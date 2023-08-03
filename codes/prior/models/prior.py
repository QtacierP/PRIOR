from prior.models.base import PretrainModel
from prior.modules.spb import SPB
from prior.modules.local_attention import LocalCrossAttention
from prior.modules.sentence_pool import SentenceAttentionPool
from prior.decoders.sentence import CrossModalityBertDecoder
from prior.modules.gather import SentenceGather
from prior.decoders.image import ImageDecoder
from torch.utils.data import DataLoader
from pl_bolts.models.self_supervised.simclr.simclr_module import SyncFunction
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pl_bolts.optimizers.lars import LARS

class Prior(PretrainModel):
    task='pretrain'
    def __init__(self, text_encoder, image_encoder, gpus,  stage1_epochs=20, stage2_epochs=30, stage3_epochs=50, max_epochs=100, stage1_warmup_epochs=1, stage2_warmup_epochs=1, stage3_warmup_epochs=5, batch_size=16, optim='adam', scheduler='linear_warmup_cosine_annealing', stage1_learning_rate=1e-5, stage1_learning_rate_start=1e-7, stage1_learning_rate_end=0, stage1_weight_decay=1e-6,  stage2_learning_rate=1e-5, stage2_learning_rate_start=1e-7, stage2_learning_rate_end=0, stage2_weight_decay=1e-6, stage3_learning_rate=5e-6, stage3_learning_rate_start=1e-8, stage3_learning_rate_end=0, stage3_weight_decay=1e-6, 
    temperature=0.1, local_temperature=0.1, embed_dim=512, image_rec_drop_out_rate=0.5, spb_k=512, num_queries=16, gahter_pool='avg', lambda_proto=10, exclude_bn_bias = False, train_dataset=None, validation_dataset=None, num_workers=0, temp_decay='fixed', frozen_text_encoder=False, ckpt_path='checkpoints/'):
        super().__init__(text_encoder=text_encoder, image_encoder=image_encoder)

        # Get embedding space from vision/language model
        self.vision_width =  image_encoder.get_width()
        self.text_width = text_encoder.get_width()
        self.embed_dim = embed_dim

        # Define text global pooling over sentences
        self.global_text_attention = SentenceAttentionPool(16, embed_dim, pos_embed=False) # Max sentence num: 32
        
        # Define project 
        self.local_vision_width = image_encoder.get_local_width()
        self.local_text_width = text_encoder.get_width()
        self.global_image_projection = nn.Linear(self.vision_width, self.embed_dim)
        self.local_image_projection = nn.Linear(image_encoder.get_local_width(), self.embed_dim)
        self.global_text_projection =  nn.Linear(self.text_width, self.embed_dim)
        self.local_text_projection =  nn.Linear(self.text_width, self.embed_dim)
        self.predictor = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2),
                                        nn.ReLU(inplace=True), # hidden layer 
                                        nn.Linear(embed_dim // 2, embed_dim)) # output layer # used for simsiam loss


        # Define decoders  
        self.num_queries = num_queries
        self.sentence_decoder = CrossModalityBertDecoder() # used for sentence prototype reconstruction
        self.prototype_queries = nn.Parameter(torch.randn(1, num_queries, embed_dim)) # input of sentece decoder
        self.image_decoder = ImageDecoder(embed_dim * 2, encoder_name=image_encoder.get_name(), image_dropout=image_rec_drop_out_rate)

        # Define local-interaction 
        self.local_cross_attention =  LocalCrossAttention(embed_dim)

        # Define temp for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.local_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / local_temperature))

        # Define SPB
        self.sentence_bank = SPB(embed_dim, spb_k)
        
        # Define hyper-params for optimization
        self.exclude_bn_bias = exclude_bn_bias
        self.batch_size = batch_size
        self.optim = optim
        self.scheduler = scheduler
        self.stage1_warmup_epochs = stage1_warmup_epochs
        self.stage1_learning_rate = stage1_learning_rate
        self.stage1_learning_rate_start = stage1_learning_rate_start
        self.stage1_learning_rate_end = stage1_learning_rate_end
        self.stage1_weight_decay = stage1_weight_decay
        self.stage2_warmup_epochs = stage2_warmup_epochs
        self.stage2_learning_rate = stage2_learning_rate
        self.stage2_learning_rate_start = stage2_learning_rate_start
        self.stage2_learning_rate_end = stage2_learning_rate_end
        self.stage2_weight_decay = stage2_weight_decay
        self.stage3_warmup_epochs = stage3_warmup_epochs
        self.stage3_learning_rate = stage3_learning_rate
        self.stage3_learning_rate_start = stage3_learning_rate_start
        self.stage3_learning_rate_end = stage3_learning_rate_end
        self.stage3_weight_decay = stage3_weight_decay
        
        self.max_epochs = stage1_epochs +  stage2_epochs +  stage3_epochs
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs =  stage2_epochs
        self.stage3_epochs = stage3_epochs

        # Define loss hyper-params
        self.lambda_proto = lambda_proto
        self.temp_decay = temp_decay

        # Define NLP gather 
        self.item_gather = SentenceGather(gahter_pool, embed_dim)

        # cache for loss
        self.last_local_batch_size = None
        self.global_alignment_labels = None
        
        # Define dataset 
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.num_workers = num_workers
        self.train_iters_per_epoch = len(self.train_dataset)  // ( len(gpus) * batch_size)

        # for dist-training, log...
        self.gpus = gpus
        self.ckpt_path = ckpt_path
        
        # freeze/finetuning params
        if frozen_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        self.inititalize_parameters()
        
    def inititalize_parameters(self):
        # Initialize parameters
        nn.init.normal_(self.global_image_projection.weight, std=self.vision_width ** -0.5)
        nn.init.normal_(self.global_text_projection.weight, std=self.text_width ** -0.5)
        nn.init.normal_(self.local_image_projection.weight, std=self.local_vision_width ** -0.5)
        nn.init.normal_(self.local_text_projection.weight, std=self.local_text_width ** -0.5)
        nn.init.normal_(self.predictor[0].weight, std=self.embed_dim ** -0.5)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self) :
        if self.validation_dataset is not None:
            return DataLoader(self.validation_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def encode_image(self, image):
        local_image_features, global_image_features, image_features_list = self.image_encoder(image, return_features=True)
        return self.local_image_projection(local_image_features), self.global_image_projection(global_image_features)
    
    def encode_text(self, text):
        x = self.text_encoder(text)
        local_text_features = x['last_hidden_state'] 
        local_text_features = local_text_features  
        global_text_features = x['pooler_output'] # Although we get the global features, we do not use it 
        return self.local_text_projection(local_text_features), global_text_features

    def global_alignment_loss(self, image_embed, text_embed):
        # SimCLR style loss
        logit_scale = self.logit_scale.exp()
        local_batch_size = image_embed.size(0)
        if local_batch_size != self.last_local_batch_size:
            self.global_alignment_labels = local_batch_size * self.local_rank + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather features from all GPUs
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            image_embed_all= SyncFunction.apply(image_embed)
            text_embed_all= SyncFunction.apply(text_embed)
        else:
            image_embed_all= image_embed
            text_embed_all= text_embed
        
        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()
        image_loss = F.cross_entropy(logits_per_image, self.global_alignment_labels)
        text_loss = F.cross_entropy(logits_per_text, self.global_alignment_labels)
        loss = (image_loss + text_loss) / 2
        return {'global_alignment_loss': loss}

    def local_alignment_loss(self, local_image_embed_stacks, local_text_embed_stacks):
        total_image_loss = 0.
        total_text_loss = 0.
        text_to_local_image_embed_stacks = []
        # TODO: maybe we can optimize this step ?
        # get each instance 
        for idx in range(local_image_embed_stacks.size(0)):
            local_text_embed = local_text_embed_stacks[idx] # get sentence-level representation 
            local_image_embed = local_image_embed_stacks[idx] # get patch-level representation
            text_to_local_image_embed, text_to_local_image_atten, image_to_local_text_embed, image_to_local_text_atten  = self.local_cross_attention(local_image_embed, local_text_embed) 
            # for local text-to-image alignment, we employ the simsiam loss without negative sample 
            image_loss = self.simsiam_loss_func(local_image_embed, text_to_local_image_embed, self.predictor)
            # for local image-to-text alignment, we just use the contrastive loss
            text_loss = self.text_local_loss_fn(local_text_embed, image_to_local_text_embed)
            total_image_loss += image_loss
            total_text_loss += text_loss
            text_to_local_image_embed_stacks.append(text_to_local_image_embed.unsqueeze(0))
        # concatenate the text-to-image features to assist image reconstruction (under text condition)
        self.text_to_local_image_embed_stacks = torch.cat(text_to_local_image_embed_stacks, dim=0)
        return {'local_image_loss': total_image_loss / local_image_embed_stacks.size(0), 'local_text_loss': total_text_loss / local_image_embed_stacks.size(0)}


    def text_local_loss_fn(self, embed_A, embed_B, norm=True):
        '''
        Similarly to CUT[1], we only utilized internal negative samples in a single report. 
        Although incorporating additional negative sentences from other patients could potentially provide more negative samples, we observed a decline in performance. This outcome is understandable, as different reports may contain highly similar sentences (especially for normal sample).
        [1] Park T, Efros A A, Zhang R, et al. Contrastive learning for unpaired image-to-image translation[C]//Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IX 16. Springer International Publishing, 2020: 319-345.
        '''
        logit_scale = self.local_logit_scale.exp()
        if norm:
            embed_A = F.normalize(embed_A, dim=-1, p=2)
            embed_B = F.normalize(embed_B, dim=-1, p=2)
        self.lc_labels = torch.arange(embed_B.size(0), device=embed_B.device).long()
        logits_per_image = logit_scale * embed_B @ embed_A.t()
        logits_per_text = logit_scale * embed_A @ embed_B.t()
        image_loss = F.cross_entropy(logits_per_image, self.lc_labels)
        text_loss = F.cross_entropy(logits_per_text, self.lc_labels)
        loss = (image_loss + text_loss) / 2   
        return loss


    def simsiam_loss(self, image_to_local_text_embed, text_to_local_image_embed, local_image_embed, local_text_embed): 
        '''
        The convolutions in encoder may cause overlap between the receptive fields of the patches, a simple negative sampling strategy is not applicable.
        '''
        text_loss = self.simsiam_loss_func(image_to_local_text_embed, local_text_embed, self.predictor, flag='text')
        image_loss = self.simsiam_loss_func(text_to_local_image_embed, local_image_embed, self.predictor, flag='image')
        return  image_loss, text_loss    

    def simsiam_loss_func(self, x, y, predictor, flag='image'):
        p_x = predictor(x)
        p_y = predictor(y)
        z_x = x.detach()
        z_y = y.detach()
        return - (F.cosine_similarity(p_x, z_y, dim=-1).mean() + F.cosine_similarity(p_y, z_x, dim=-1).mean()) * 0.5
   
    def recon_image(self, image_embed, image):
        # reshape the cross-modal features to the same shape as image
        self.text_to_local_image_embed_stacks = self.text_to_local_image_embed_stacks.view(-1, self.text_to_local_image_embed_stacks.size(-1), *self.image_encoder.get_last_spatial_info())
        # reshape the image features to the [B, C, H, W]
        image_embed = image_embed.view(-1, image_embed.size(-1), *self.image_encoder.get_last_spatial_info())
        output = self.image_decoder(image_embed, self.text_to_local_image_embed_stacks, image)
        return {'rec_image_loss': output['loss']}
     

    def recon_report(self, image_embed, padding_local_text_embed_stacks, padding_proto_local_text_embed_stacks, padding_embed_ind_stacks, sentence_attention_mask, global_text_embed):
        '''
        image_embed: the local embedding of image, [b, i, d]
        padding_local_text_embed_stacks: the local embedding of text, [b, n, d]
        padding_proto_local_text_embed_stacks: the local embedding of prototype text, [b, n, d]
        padding_embed_ind_stacks: the index of local text embedding in SPB, [b, n]
        sentence_attention_mask: the padding mask of text, [b, n]
        global_text_embed: the global embedding of text, [b, d]
        '''
        # reshape the queries to the batch size 
        input_queies = self.prototype_queries.repeat(image_embed.size(0), 1, 1)
        logits = self.sentence_decoder(input_queies, image_embed)
        # Sentence Prototype Generation loss
        recon_report_loss_dict, logits_stacks = self.spg_loss(logits, padding_local_text_embed_stacks, padding_proto_local_text_embed_stacks, padding_embed_ind_stacks, sentence_attention_mask)
        # query the genereated prototypes from SPB
        proto_logits_stacks, _,  logits_embed_ind = self.sentence_bank(torch.cat(logits_stacks, dim=0))
        proto_logits_stacks = self.rec_text_stacks(proto_logits_stacks, logits_stacks)
        logits_global_text_embed = self.get_global_text_representation(proto_logits_stacks)
        # global prediction alignment
        logits_global_alignment_dict = self.global_alignment_loss(logits_global_text_embed, global_text_embed)
        recon_report_loss_dict['gpa_loss'] = logits_global_alignment_dict['global_alignment_loss'] 
        return recon_report_loss_dict


    def sim_matrix(self, a, b, norm_dim=1, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=norm_dim)[:, None], b.norm(dim=norm_dim)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def matching_label(self, outputs, targets, norm=False, metric='l1'):
        # use the hungarian algorithm to match the labels
        if norm:
            outputs = F.normalize(outputs, dim=-1)
        if metric == 'l1':
            cost = torch.cdist(outputs, targets, p=1).detach().cpu()
        elif metric == 'l2':
            cost = torch.cdist(outputs, targets, p=2).detach().cpu()
        elif metric == 'cosine':
            cost = -self.sim_matrix(outputs, targets).detach().cpu()
        indices = linear_sum_assignment(cost)  
        return indices

    def spg_loss(self, logits, features, protos, proto_index, sentence_masks):
        '''
        logits: the output of the sentence decoder (reconstructed sentence prototype), [B, N, D]
        features: the sentence features [before SPB], [B, N, D]
        protos: the sentence prototype [after SPB], [B, N, D]
        proto_indexs: the sentence prototype index, [B, N]
        sentence_masks: the sentence mask, [B, N]
        '''
        spg_loss = 0.
        total_proto_index = []
        total_logits = []
        total_features = []
        logits_stacks = []
        total_label = []
        # TODO: optimize the matching process [use padding prototype to parallel the matching process ?]
        for logit, feature, proto, mask, label in zip(logits, features, protos, sentence_masks, proto_index): 
            # use hungarian algorithm to match the logit and the prototype
            inds = self.matching_label(logit, proto)
            # rearrange the logit according to the matching label
            rearrange_logit = logit[inds[1]]
            # exclude the padding token [mask=0]
            length = mask.sum().int()
            roi_label = label[:length].int()
            roi_logit = rearrange_logit[:length] 
            roi_feature = feature[:length]
            roi_proto = proto[:length]
            # calculate the spg loss
            spg_loss += F.l1_loss(rearrange_logit, proto)
            # append the roi logit and roi feature
            total_label.append(roi_label)
            total_logits.append(roi_logit)
            total_features.append(roi_feature)
            logits_stacks.append(roi_logit)
        total_label = torch.cat(total_label, dim=0)
        total_logits = torch.cat(total_logits, dim=0)
        total_features = torch.cat(total_features, dim=0)
        # calculate the spg loss 
        spg_loss = spg_loss / len(logits)
        # calculate the kl_loss
        # kl_loss is aimed to maintain the query consistency between logits and features
        kl_loss = self.sentence_bank.cal_loss(total_logits) 
        return {'spg_loss': spg_loss * self.lambda_proto, 'kl_loss': kl_loss}, logits_stacks
    

    def padding_sentence_stacks(self, sentence_stacks, max_length=16):
        # padding sentence_stacks to the same length
        # sentence_stacks: list 
        # length: int
        # return: [B, length, D] tensor, attention mask [B, length]
        batch_size = len(sentence_stacks)
        padded_sentence_stacks = torch.zeros(batch_size, max_length, self.embed_dim).to(sentence_stacks[0].device)
        #trancated_sentence_stacks = []
        attention_mask = torch.zeros(batch_size, max_length).to(sentence_stacks[0].device)
        for i, sentence_stack in enumerate(sentence_stacks):
            if len(sentence_stack) > max_length:
                padded_sentence_stacks[i, :] = sentence_stack[:max_length]
                attention_mask[i, :] = 1
                #trancated_sentence_stacks.append(sentence_stack[:max_length])
            else:
                padded_sentence_stacks[i, :len(sentence_stack)] = sentence_stack
                attention_mask[i, :len(sentence_stack)] = 1
                #trancated_sentence_stacks.append(sentence_stack)
        return padded_sentence_stacks, attention_mask
       
    def get_global_text_representation(self, local_text_embed_stacks):
        batch_stacks = []
        for local_text_embed in local_text_embed_stacks:
            batch_stacks.append(self.global_text_attention(local_text_embed.unsqueeze(dim=0)))
        return torch.cat(batch_stacks, dim=0)

    def trancated_sentence_stack(self, stacks, max_length=16):
        # truncate sentence_stack to max_length
        # stacks: list of [L, D]
        # return: [B, length, D]
        truncated_stacks = []
        for i, stack in enumerate(stacks):
            if len(stack) > max_length:
                truncated_stacks.append(stack[:max_length])
            else:
                truncated_stacks.append(stack)
        return truncated_stacks
    
    def padding_embed_ind_stacks(self, embed_ind, max_length=16):
        batch_size = len(embed_ind)
        # TODO: change the -1 flag to padding token index, which may speed up the matching process
        padded_embed_ind = torch.ones(batch_size, max_length).to(embed_ind[0].device) * -1 # -1 flag for debugging
        for i, embed_ind_stack in enumerate(embed_ind):
            if len(embed_ind_stack) > max_length:
                padded_embed_ind[i, :] = embed_ind_stack[:max_length]
            else:
                padded_embed_ind[i, :len(embed_ind_stack)] = embed_ind_stack
        return padded_embed_ind

    def rec_text_stacks(self, flatten_stacks, stacks, flatten_embed_ind=None):
        new_stacks = []
        new_embed_ind = []
        idx = 0
        for sample in stacks:
            new_stacks.append(flatten_stacks[idx: idx + sample.size(0)])
            if flatten_embed_ind is not None:
                new_embed_ind.append(flatten_embed_ind[idx: idx + sample.size(0)])
            idx += sample.size(0)
        #print('len of sentence_stacks', idx)
        if flatten_embed_ind is not None:
            return new_stacks, new_embed_ind
        return new_stacks


    def stage1_step(self, batch, stage='stage1'):
        '''
        Stage 1 only employs the alignment wihout SPB, in order to speed up the convergence and avoid unstable training
        '''
        image = batch['image']
        text = batch['text']
        '''
        =================================================================
        Encode image and text and get the local and global representation
        =================================================================
        '''
        # Embed image
        local_image_embed, global_image_embed = self.encode_image(image)
        # Embed text
        local_text_embed, _ = self.encode_text(text)
        # gather local text embedding on sentence level
        local_text_embed_stacks = self.item_gather(local_text_embed, batch)
        # get global text embedding
        global_text_embed = self.get_global_text_representation(local_text_embed_stacks)
        '''
        =================================================================
        Calculate the alignment loss
        =================================================================
        '''
        # local alignment loss (w.o. SPB)
        local_loss_dict = self.local_alignment_loss(local_image_embed, local_text_embed_stacks) # shared local image embedding
        # global contrastive loss
        global_alignment_dict = self.global_alignment_loss(global_image_embed, global_text_embed)  
        '''
        =================================================================
        Log the loss
        =================================================================
        '''
        loss_dict = {}
        loss_dict[stage + '_loss'] = 0
        for k, v in global_alignment_dict.items():
            loss_dict[stage + '_' + k] = v
            if 'loss' in k:   
                loss_dict[stage + '_loss'] += v
        for k, v in local_loss_dict.items():
            loss_dict[stage + '_' + k] = v
            if 'loss' in k:
                loss_dict[stage + '_loss'] += v
        return local_image_embed, loss_dict


    def stage2_step(self, batch, stage='stage2'):
        '''
        Stage 2 employs the alignment with SPB
        '''
        image = batch['image']
        text = batch['text']
        '''
        =================================================================
        Encode image and text and get the local and global representation
        In stage 2, we need SPB to get the sentence prototype
        =================================================================
        '''
        # Embed image
        local_image_embed, global_image_embed = self.encode_image(image)
        # Embed text
        local_text_embed, _ = self.encode_text(text)
        # gather local text embedding on sentence level
        local_text_embed_stacks = self.item_gather(local_text_embed, batch)
        # Query sentence prototype  from SPB
        proto_local_text_embed_stacks, proto_loss, embed_ind = self.sentence_bank(torch.cat(local_text_embed_stacks, dim=0))
        # rec stacks according to the number of sentences in each sample
        proto_local_text_embed_stacks = self.rec_text_stacks(proto_local_text_embed_stacks, local_text_embed_stacks)
        # get global text embedding 
        global_text_embed = self.get_global_text_representation(proto_local_text_embed_stacks)
        '''
        =================================================================
        Calculate the alignment loss (LAM)
        In stage 2, the local alignment loss is calculated based on the sentence prototype
        =================================================================
        '''
        # local alignment loss
        local_loss_dict = self.local_alignment_loss(local_image_embed, proto_local_text_embed_stacks) 
        # global contrastive loss
        global_alignment_dict = self.global_alignment_loss(global_image_embed, global_text_embed)
        '''
        =================================================================
        Log the loss
        =================================================================
        '''
        loss_dict = {}
        loss_dict[stage + '_loss'] = 0
        for k, v in global_alignment_dict.items():
            loss_dict[stage + '_' + k] = v
            if 'loss' in k:   
                loss_dict[stage + '_loss'] += v 
        for k, v in local_loss_dict.items():
            loss_dict[stage + '_' + k] = v
            if 'loss' in k:
                loss_dict[stage + '_loss'] += v
        loss_dict[stage + '_proto_loss_'] = proto_loss * self.lambda_proto
        loss_dict[stage + '_loss'] += proto_loss * self.lambda_proto
        return local_image_embed, loss_dict



    def stage3_step(self, batch, stage='stage3'):
        '''
        Stage 3 employs the alignment with SPB and CCR
        '''
        image = batch['image']
        text = batch['text']
        '''
        =================================================================
        Encode image and text and get the local and global representation
        In stage 3, we need SPB to get the sentence prototype
        =================================================================
        '''
        # Embed image
        local_image_embed, global_image_embed = self.encode_image(image)
        # Embed text
        local_text_embed, _ = self.encode_text(text)
        # gather local text embedding on sentence level
        local_text_embed_stacks = self.item_gather(local_text_embed, batch)
        # for CCR, we need to limit the max length of report (no longer than num_queries)
        trancated_local_text_embed_stacks = self.trancated_sentence_stack(local_text_embed_stacks, self.num_queries) 
        # get prototype
        proto_local_text_embed_stacks, proto_loss, embed_ind = self.sentence_bank(torch.cat(trancated_local_text_embed_stacks, dim=0))
         # rec stacks & embedding idx according to the number of sentences in each sample
        proto_local_text_embed_stacks, embed_ind_stacks  = self.rec_text_stacks(proto_local_text_embed_stacks, trancated_local_text_embed_stacks, embed_ind)
        '''
        =================================================================
        Since the num_queries is fixed
        we need to pad the local text stacks and prototype stacks to the same length
        =================================================================
        '''
        # padding local text stacks & protos to the same length
        padding_local_text_embed_stacks, sentence_attention_mask = self.padding_sentence_stacks(trancated_local_text_embed_stacks, self.num_queries)
        padding_proto_local_text_embed_stacks, sentence_attention_mask = self.padding_sentence_stacks(proto_local_text_embed_stacks, self.num_queries)
        padding_embed_ind_stacks = self.padding_embed_ind_stacks(embed_ind_stacks, self.num_queries)
        '''
        =================================================================
        Calculate the alignment loss (LAM)
        In stage 3, the local alignment loss is calculated based on the sentence prototype
        =================================================================
        '''
        # get global text embed
        global_text_embed = self.get_global_text_representation(proto_local_text_embed_stacks)
        # global contrastive loss
        global_alignment_dict = self.global_alignment_loss(global_image_embed, global_text_embed)
        # local alignment loss
        local_loss_dict = self.local_alignment_loss(local_image_embed, proto_local_text_embed_stacks)
        '''
        =================================================================
        Conditional Cross-modality reconstruction loss (CCR)
        =================================================================
        '''
        # Reconstruct image
        rec_image_loss_dict = self.recon_image(local_image_embed, image)

        # Reconstruct report prototype
        rec_report_loss_dict = self.recon_report(local_image_embed, padding_local_text_embed_stacks, padding_proto_local_text_embed_stacks, padding_embed_ind_stacks, sentence_attention_mask, global_text_embed)
        '''
        =================================================================
        Log the loss
        =================================================================
        '''
        loss_dict = {}
        loss_dict[stage + '_loss'] = 0
        for k, v in global_alignment_dict.items():
            loss_dict[stage + '_' + k] = v
            if 'loss' in k:   
                loss_dict[stage + '_loss'] += v * 10 # need tune hyper-parameters for stage 3
        for k, v in local_loss_dict.items():
            loss_dict[stage + '_' + k] = v
            if 'loss' in k:
                loss_dict[stage + '_loss'] += v
        for k, v in rec_image_loss_dict.items():
            loss_dict[stage + '_' + k] = v
            if 'loss' in k: 
                loss_dict[stage + '_loss'] += v
        for k, v in rec_report_loss_dict.items():
            loss_dict[stage + '_' + k] = v
            if 'loss' in k: 
                loss_dict[stage + '_loss'] += v
        loss_dict[stage + '_proto_loss_'] = proto_loss * self.lambda_proto
        loss_dict[stage + '_loss'] += proto_loss * self.lambda_proto
        return local_image_embed, loss_dict
        
        
    def forward(self, batch):
        image_features  = self.image_encoder(batch)
        text_features = self.text_encoder(batch)
        return image_features, text_features

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == 0:
            # Stage 1 starts
            optimizers, lr_schedulers, _  = self.call_optimization(max_epochs=self.stage1_epochs, warmup_epochs=self.stage1_warmup_epochs, weight_decay=self.stage1_weight_decay, learning_rate=self.stage1_learning_rate, learning_rate_start=self.stage1_learning_rate_start, learning_rate_end=self.stage1_learning_rate_end)
            self.trainer.lr_schedulers = lr_schedulers
            self.trainer.optimizers = optimizers
        elif self.current_epoch == self.stage1_epochs: 
            optimizers, lr_schedulers, _  = self.call_optimization(max_epochs=self.stage2_epochs, warmup_epochs=self.stage2_warmup_epochs, weight_decay=self.stage2_weight_decay, learning_rate=self.stage2_learning_rate, learning_rate_start=self.stage2_learning_rate_start, learning_rate_end=self.stage2_learning_rate_end)
            self.trainer.lr_schedulers = lr_schedulers
            self .trainer.optimizers = optimizers
        elif self.current_epoch == self.stage1_epochs + self.stage2_epochs:
            optimizers, lr_schedulers, _  = self.call_optimization(max_epochs=self.stage3_epochs, warmup_epochs=self.stage3_warmup_epochs, weight_decay=self.stage3_weight_decay, learning_rate=self.stage3_learning_rate, learning_rate_start=self.stage3_learning_rate_start, learning_rate_end=self.stage3_learning_rate_end)
            self.trainer.lr_schedulers = lr_schedulers
            self .trainer.optimizers = optimizers

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx) -> None:
        if self.current_epoch >= self.stage1_epochs:
            current_step = self.global_step - self.stage1_epochs * self.train_iters_per_epoch
            total_step = (self.stage2_epochs + self.stage3_epochs) * self.train_iters_per_epoch
            self.sentence_bank.set_temp(current_step, total_step, self.temp_decay)
        self.log('spb_temp', self.sentence_bank.curr_temp, on_step=True, on_epoch=False, prog_bar=True, logger=True)
       
    def on_train_epoch_end(self) -> None:
        if self.current_epoch == self.stage1_epochs - 1: 
            if self.global_rank == 0:      
                self.trainer.save_checkpoint(f"{self.ckpt_path}/stage2_start.ckpt")
        if self.current_epoch == self.stage1_epochs + self.stage2_epochs - 1: 
            if self.global_rank == 0:      
                self.trainer.save_checkpoint(f"{self.ckpt_path}/stage3_start.ckpt")

    def call_optimization(self, max_epochs=None, warmup_epochs=None, learning_rate=None, learning_rate_start=None, learning_rate_end=None, weight_decay=None, slow_text_encoder=False):
        optim_conf = self.configure_optimizers(max_epochs=max_epochs, warmup_epochs=warmup_epochs, slow_text_encoder=slow_text_encoder, learning_rate=learning_rate, learning_rate_start=learning_rate_start, learning_rate_end=learning_rate_end, weight_decay=weight_decay)
        optimizers, lr_schedulers, optimizer_frequencies, monitor = self.trainer._configure_optimizers(optim_conf)
        lr_schedulers = self.trainer._configure_schedulers(lr_schedulers, monitor, not self.automatic_optimization)
        return optimizers, lr_schedulers, optimizer_frequencies


    def training_step(self, batch, batch_idx):
        if self.current_epoch < self.stage1_epochs:
            image_feaures, loss_dict = self.stage1_step(batch)
            loss = loss_dict['stage1_loss']
        elif self.current_epoch >= self.stage1_epochs and self.current_epoch < self.stage2_epochs + self.stage1_epochs:
            image_feaures, loss_dict = self.stage2_step(batch)
            loss = loss_dict['stage2_loss']
        elif self.current_epoch >= self.stage2_epochs + self.stage1_epochs:
            image_feaures, loss_dict = self.stage3_step(batch)
            loss = loss_dict['stage3_loss']
        self.log_dict(loss_dict, on_step=True, on_epoch=False, prog_bar=True)
        return loss


    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=["bias", "bn"]):
        params = []
        excluded_params = []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
        ]


    def exclude_from_text_encoder(self, named_params, weight_decay):
        # exclude discriminator param
        params = []
        excluded_params = []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif 'text_encoder' in name:
                excluded_params.append(param)
            else:
                params.append(param)
        return params, excluded_params
        

    def configure_optimizers(self, learning_rate=1e-5, learning_rate_start=1e-7, learning_rate_end=0, max_epochs=100, warmup_epochs=1, slow_text_encoder=False, weight_decay=1e-6):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=weight_decay)
        else:
            params = self.parameters()
        if slow_text_encoder:
            other_params, text_params = self.exclude_from_text_encoder(self.named_parameters(), weight_decay=weight_decay)
            params = [{"params": text_params}, {"params": other_params}]
        if self.optim == "lars":
            optimizer = LARS(
                params,
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif self.optim == "adamw":
            optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        if slow_text_encoder:  
            optimizer.param_groups[0]['lr'] = learning_rate / 10 # slow down text encoder
            optimizer.param_groups[1]['lr'] = learning_rate
        warmup_steps = self.train_iters_per_epoch * warmup_epochs 
        total_steps = self.train_iters_per_epoch * max_epochs
        if self.scheduler == 'cosine_warmup_linear_annealing':
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    linear_warmup_decay(warmup_steps, total_steps, cosine=True),
                ),
                "interval": "step",
                "frequency": 1,
            }
        elif self.scheduler == 'linear_warmup_cosine_annealing':
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=warmup_steps,
                    max_epochs=total_steps,
                    warmup_start_lr=learning_rate_start, eta_min=learning_rate_end),
                "interval": "step",
                "frequency": 1,
            }
        elif self.scheduler == 'cosine_decay':
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps),
                "interval": "step",
                "frequency": 1,
            }
        return [optimizer], [scheduler]

        






        