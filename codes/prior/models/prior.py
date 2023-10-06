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
    def __init__(self, text_encoder, image_encoder, gpus,  max_epochs=20, warmup_epochs=2, batch_size=16, optim='adam', scheduler='linear_warmup_cosine_annealing', learning_rate=1e-4, learning_rate_start=1e-6, learning_rate_end=1e-5, weight_decay=1e-6, temperature=0.1, local_temperature=0.1, embed_dim=512, exclude_bn_bias = False, train_dataset=None, validation_dataset=None, num_workers=0, frozen_text_encoder=False, ckpt_path='checkpoints/'):
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
        self.global_image_projection = nn.Sequential(nn.Linear(self.vision_width, self.embed_dim))
        self.local_image_projection = nn.Sequential(nn.Linear(image_encoder.get_local_width(), self.embed_dim))
        self.global_text_projection =  nn.Sequential(nn.Linear(self.text_width, self.embed_dim))
        self.local_text_projection =  nn.Sequential(nn.Linear(self.text_width, self.embed_dim))
        self.predictor = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(embed_dim // 2, embed_dim)) # output layer # used for simsiam 


        # Define decoders  
        self.sentence_decoder = CrossModalityBertDecoder() # used for sentence prototype reconstruction
        self.prototype_queries = nn.Parameter(torch.randn(1, 16, embed_dim)) # input of sentece decoder
        self.image_decoder = ImageDecoder(embed_dim * 2, encoder_name='resnet50', image_dropout=0.5)

        # Define local-interaction 
        self.local_cross_attention =  LocalCrossAttention(embed_dim)

        # Define temp for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.local_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / local_temperature))

        # Define SPB
        self.sentence_bank = SPB(embed_dim, 512)
        
        # Define hyper-params for optimization
        self.exclude_bn_bias = exclude_bn_bias
        self.batch_size = batch_size
        self.optim = optim
        self.scheduler = scheduler
        self.learning_rate = learning_rate
        self.learning_rate_start = learning_rate_start
        self.learning_rate_end = learning_rate_end
        self.weight_decay= weight_decay
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.stage1_epochs = int(max_epochs * 0.2)
        self.stage2_epochs = int(max_epochs * 0.3)
        self.stage3_epochs = max_epochs - self.stage1_epochs - self.stage2_epochs

        # Define NLP gather
        self.item_gather = SentenceGather()

        # cache for loss
        self.last_local_batch_size = None
        self.global_alignment_labels = None
        
        # Define dataset 
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.num_workers = num_workers
        self.train_iters_per_epoch = len(self.train_dataset)  // ( len(gpus) * batch_size)

        # for dis-training, log...
        self.gpus = gpus
        self.ckpt_path = ckpt_path
        
        # tuning and params
        if frozen_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        self.inititalize_parameters()
        
    def inititalize_parameters(self):
        nn.init.normal_(self.global_image_projection[0].weight, std=self.vision_width ** -0.5)
        nn.init.normal_(self.global_text_projection[0].weight, std=self.text_width ** -0.5)
        nn.init.normal_(self.local_image_projection[0].weight, std=self.local_vision_width ** -0.5)
        nn.init.normal_(self.local_text_projection[0].weight, std=self.local_text_width ** -0.5)
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
        global_text_features = x['pooler_output'] 
        return self.local_text_projection(local_text_features), self.global_text_projection(global_text_features)

    def global_alignment_loss(self, image_embed, text_embed, logit_scale):
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
        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.global_alignment_labels).sum()
            acc = 100 * correct / local_batch_size
        return {'global_alignment_loss': loss, 'global_alignment_acc': acc}

    def local_alignment_loss(self, local_image_embed_stacks, local_text_embed_stacks):

        total_image_loss = 0.
        total_text_loss = 0.
        text_to_local_image_embed_stacks = []
        for idx in range(local_image_embed_stacks.size(0)):
            local_text_embed = local_text_embed_stacks[idx]
            local_image_embed = local_image_embed_stacks[idx]
            text_to_local_image_embed, text_to_local_image_atten, image_to_local_text_embed, image_to_local_text_atten  = self.cross_attention(local_image_embed, local_text_embed) # 
            image_loss = self.simsiam_loss_func(local_image_embed, text_to_local_image_embed, self.predictor)
            text_loss = self.text_local_loss_fn(local_text_embed, image_to_local_text_embed, self.local_logit_scale.exp())
            total_image_loss += image_loss
            total_text_loss += text_loss
            text_to_local_image_embed_stacks.append(text_to_local_image_embed.unsqueeze(0))
        self.text_to_local_image_embed_stacks = torch.cat(text_to_local_image_embed_stacks, dim=0)
        return {'local_image_loss': total_image_loss / local_image_embed_stacks.size(0), 'local_text_loss': total_text_loss / local_image_embed_stacks.size(0)}


    def text_local_loss_fn(self, embed_A, embed_B, logit_scale, norm=True):
        if norm:
            embed_A = F.normalize(embed_A, dim=-1, p=2)
            embed_B = F.normalize(embed_B, dim=-1, p=2)
        embed_A_all= embed_A
        embed_B_all= embed_B
        self.lc_labels = torch.arange(embed_B.size(0), device=embed_B.device).long()
        #self.lc_last_local_batch_size = local_batch_size 
        logits_per_image = logit_scale * embed_B @ embed_A_all.t()
        logits_per_text = logit_scale * embed_A @ embed_B_all.t()
        image_loss = F.cross_entropy(logits_per_image, self.lc_labels)
        text_loss = F.cross_entropy(logits_per_text, self.lc_labels)
        loss = (image_loss + text_loss) / 2   
        return loss


    def cross_attention(self, source, target, mask=None):
        t2s_output, attention, s2t_output, attention_1 = self.local_cross_attention(source, target)
        return t2s_output, attention, s2t_output, attention_1

    def simsiam_loss(self, image_to_local_text_embed, text_to_local_image_embed, local_image_embed, local_text_embed): 
        text_loss = self.simsiam_loss_func(image_to_local_text_embed, local_text_embed, self.predictor, flag='text')
        image_loss = self.simsiam_loss_func(text_to_local_image_embed, local_image_embed, self.predictor, flag='image')
        return  image_loss, text_loss    

    def simsiam_loss_func(self, x, y, predictor, flag='image'):
        p_x = predictor(x)
        p_y = predictor(y)
        z_x = x.detach()
        z_y = y.detach()
        return - (F.cosine_similarity(p_x, z_y, dim=-1).mean() + F.cosine_similarity(p_y, z_x, dim=-1).mean()) * 0.5
   
    def recon_image(self, image_embed, text_embed, image):
        self.text_to_local_image_embed_stacks = self.text_to_local_image_embed_stacks.view(-1, self.text_to_local_image_embed_stacks.size(-1), 7 ,7)
        image_embed = image_embed.view(-1, image_embed.size(-1), 7, 7)
        output = self.image_decoder(image_embed, self.text_to_local_image_embed_stacks, image)
        return output['loss']
     

    def multi_modal_forward(self, image_embed):
        input_queies = self.prototype_queries.repeat(image_embed.size(0), 1, 1)
        logits = self.sentence_decoder(input_queies, image_embed)
        #selects = self.token_selector(logits)
        #logits = self.caption_classifier(logits)
        #print(logits)
        return logits


    def matching_label_l1(self, outputs, targets, norm=False):
        if norm:
            outputs = F.normalize(outputs, dim=-1)
        #cost_cosine = self.sim_matrix(outputs, targets).detach().cpu()  # batch_size x batch_size
        cost_mse = torch.cdist(outputs, targets, p=1).detach().cpu()
        #print(cost_mse)
        indices = linear_sum_assignment(cost_mse)  # indices[0] is the row indices, indices[1] is the column indices
        #print(cost_mse[indices[0], indices[1]].mean())
        return indices


    def spg_loss(self, logits, targets, q_targets, labels, sentence_mask):
        # logits: [B, L, D], L is max length of caption
        # labels: list of B x [B_L, D], B_L is the length of each item's caption
        # Solution 1: zero padding for shorter captions, using attention mask to abandon the padded captions
        # Solution 2: embed the padding into memory bank
        loss = 0.
        total_label = []
        total_logits = []
        total_targets = []
        logits_stacks = []
        for logit, target, q_target, mask, label in zip(logits, targets, q_targets, sentence_mask, labels): 
            # firstly, calculate the matching label
            #logits = F.normalize(logits, p=2, dim=-1)
            inds = self.matching_label_l1(logit, q_target)
            # then,rearrange the logit 
            rearrange_logit = logit[inds[1]]
            # rerrange the label according to the matching label and exclude the padding token
            length = mask.sum().int()
            #print(label[:length])
            roi_label = label[:length].int()
            roi_logit = rearrange_logit[:length] 
            roi_target = target[:length]
            roi_q_target = q_target[:length]
            # the fisrt loss is mse, aim to reduce the L2 distance 
            # print(F.mse_loss(rearrange_logit, q_target, reduction='none').mean(dim=-1))
            #print('l1_loss', F.l1_loss(rearrange_logit, q_target))
            #print('ce loss', F.binary_cross_entropy_with_logits(rearrange_select, mask.unsqueeze(dim=1)))
            loss += F.l1_loss(rearrange_logit, q_target) * 10 
            total_label.append(roi_label)
            total_logits.append(roi_logit)
            total_targets.append(roi_target)
            logits_stacks.append(roi_logit)
        total_label = torch.cat(total_label, dim=0)
        total_logits = torch.cat(total_logits, dim=0)
        total_targets = torch.cat(total_targets, dim=0)
        loss = loss / len(logits)
        # calculate the memory bank loss
        # memory bank loss is aimed to maintain the query consistency between logits and embedding
        memory_query_loss = self.sentence_bank.cal_loss(total_logits) 
        return {'spg_loss': loss, 'memory_query_loss': memory_query_loss}, logits_stacks
    

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

    def padding_embed_ind_stacks(self, embed_ind, max_length=16):
        batch_size = len(embed_ind)
        padded_embed_ind = torch.ones(batch_size, max_length).to(embed_ind[0].device) * -1 # -1 flag for debugging
        for i, embed_ind_stack in enumerate(embed_ind):
            if len(embed_ind_stack) > max_length:
                padded_embed_ind[i, :] = embed_ind_stack[:max_length]
            else:
                padded_embed_ind[i, :len(embed_ind_stack)] = embed_ind_stack
        return padded_embed_ind
       
    def get_global_text_representation(self, local_text_embed_stacks):
        batch_stacks = []
        for local_text_embed in local_text_embed_stacks:
            batch_stacks.append(self.global_text_attention(local_text_embed.unsqueeze(dim=0)))
        return torch.cat(batch_stacks, dim=0)

    def trancated_sentence_stack(self, stacks, max_length=16):
        # truncate sentence_stack to max_length
        # stacks: list of [L, D]
        # return: [B, length, D]
        batch_size = len(stacks)
        truncated_stacks = []
        for i, stack in enumerate(stacks):
            if len(stack) > max_length:
                truncated_stacks.append(stack[:max_length])
            else:
                truncated_stacks.append(stack)
        return truncated_stacks




    def rec_text_stacks(self, flatten_stacks, stacks):
        new_stacks = []
        idx = 0
        for sample in stacks:
            new_stacks.append(flatten_stacks[idx: idx + sample.size(0)])
            idx += sample.size(0)
        #print('len of sentence_stacks', idx)
        return new_stacks

    def rec_embed_ind_stacks(self, flatten_embed_ind, stacks):
        new_stacks = []
        idx = 0
        for sample in stacks:
            #print(sample.size())
            #print(flatten_embed_ind.size())
            embed_ind = flatten_embed_ind[idx: idx + sample.size(0)]
            # add 0 and 255 to the beginning and end of each sentence respectively
            #print(embed_ind.size())
            new_stacks.append(embed_ind)
            idx += sample.size(0)
        return new_stacks

    def rec_logits_stacks(self, logits, attention_mask):
        # logits: [B, L, D]
        # attention_mask: [B, L]
        # return: list of [L]
        batch_size = logits.size(0)
        stacks = []
        for i in range(batch_size):
            stacks.append(logits[i, attention_mask[i] == 1])
        return stacks


    def stage2_step(self, batch, stage='stage2'):
        image = batch['image']
        text = batch['text']
        # Embed image
        local_image_embed, global_image_embed = self.encode_image(image)
    
        # Embed text
        local_text_embed, _ = self.encode_text(text)
        local_text_embed_stacks = self.item_gather(local_text_embed, batch)
       
        # Query from SPB
        proto_local_text_embed_stacks, diff, embed_ind = self.sentence_bank(torch.cat(local_text_embed_stacks, dim=0))
        proto_local_text_embed_stacks = self.rec_text_stacks(proto_local_text_embed_stacks, local_text_embed_stacks)
        q_global_text_embed = self.get_global_text_representation(proto_local_text_embed_stacks)
     
        # local alignment loss
        local_loss_dict = self.local_alignment_loss(local_image_embed, proto_local_text_embed_stacks) # shared local image embedding

        # global contrastive loss
        global_alignment_dict = self.global_alignment_loss(global_image_embed, q_global_text_embed, self.logit_scale.exp())



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
        loss_dict[stage + '_text_spb_loss'] = diff * 10
        loss_dict[stage + '_loss'] += diff * 10
        return local_image_embed, loss_dict

    def stage1_step(self, batch, stage='stage1'):
        # Stage 1 only employs the alignment wihout SPB, in order to speed up the convergence
        image = batch['image']
        text = batch['text']
        # Embed image
        local_image_embed, global_image_embed = self.encode_image(image)
        # Embed text
        local_text_embed, _ = self.encode_text(text)
        local_text_embed_stacks = self.item_gather(local_text_embed, batch)
        global_text_embed = self.get_global_text_representation(local_text_embed_stacks)
        # local alignment loss
        local_loss_dict = self.local_alignment_loss(local_image_embed, local_text_embed_stacks) # shared local image embedding
        # global contrastive loss
        global_alignment_dict = self.global_alignment_loss(global_image_embed, global_text_embed, self.logit_scale.exp())  

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
 


    def stage3_step(self, batch, stage='stage3'):
        image = batch['image']
        text = batch['text']
        # Embed image
        local_image_embed, global_image_embed = self.encode_image(image)

        # Embed text
        local_text_embed, _ = self.encode_text(text)
        local_text_embed_stacks = self.item_gather(local_text_embed, batch)

        # trancate
        trancated_local_text_embed_stacks = self.trancated_sentence_stack(local_text_embed_stacks) # for CCR, we need to fix the length of report

        # get prototype
        proto_local_text_embed_stacks, diff, embed_ind = self.sentence_bank(torch.cat(trancated_local_text_embed_stacks, dim=0))
        proto_local_text_embed_stacks = self.rec_text_stacks(proto_local_text_embed_stacks, trancated_local_text_embed_stacks)
        embed_ind_stacks = self.rec_embed_ind_stacks(embed_ind, trancated_local_text_embed_stacks)

        # padding to embed ind stacks
        padding_embed_ind_stacks = self.padding_embed_ind_stacks(embed_ind_stacks)

        # get global text embed
        q_global_text_embed = self.get_global_text_representation(proto_local_text_embed_stacks)

        # global contrastive loss
        global_alignment_dict = self.global_alignment_loss(global_image_embed, q_global_text_embed, self.logit_scale.exp())

        # local alignment loss
        local_loss_dict = self.local_alignment_loss(local_image_embed, proto_local_text_embed_stacks)

        # Reconstruct image
        rec_image_loss = self.recon_image(local_image_embed, proto_local_text_embed_stacks, image)

        # Sentece prototype generation
        logits = self.multi_modal_forward(local_image_embed)
        # padding local text stacks to the same length
        padding_local_text_embed_stacks, sentence_attention_mask = self.padding_sentence_stacks(trancated_local_text_embed_stacks)
        padding_proto_local_text_embed_stacks, sentence_attention_mask = self.padding_sentence_stacks(proto_local_text_embed_stacks)
        # caption loss .
        spg_loss_dict, logits_stacks = self.spg_loss(logits, padding_local_text_embed_stacks, padding_proto_local_text_embed_stacks, padding_embed_ind_stacks, sentence_attention_mask)

        # get  aware text global embedding
        proto_logits_stacks, logits_diff,  logits_embed_ind = self.sentence_bank.query(torch.cat(logits_stacks, dim=0))
        proto_logits_stacks = self.rec_text_stacks(proto_logits_stacks, logits_stacks)
        logits_global_text_embed = self.get_global_text_representation(proto_logits_stacks)

        # global alignment loss
        logits_global_alignment_dict = self.global_alignment_loss(q_global_text_embed, logits_global_text_embed, self.logit_scale.exp())
        logits_global_alignment_loss = logits_global_alignment_dict['global_alignment_loss'] 
        logits_clip_acc = logits_global_alignment_dict['global_alignment_acc']


        loss_dict = {}
        loss_dict[stage + '_loss'] = 0
        for k, v in global_alignment_dict.items():
            loss_dict[stage + '_' + k] = v
            if 'loss' in k:   
                loss_dict[stage + '_loss'] += v * 10
        for k, v in spg_loss_dict.items():
            loss_dict[stage + '_' + k] = v
            if 'loss' in k: 
                loss_dict[stage + '_loss'] += v
        for k, v in local_loss_dict.items():
            loss_dict[stage + '_' + k] = v
            if 'loss' in k:
                loss_dict[stage + '_loss'] += v
        #loss_dict['stage' + '_image_vq_loss'] = image_diff
        loss_dict[stage + '_text_vq_loss'] = diff * 10
        loss_dict[stage + '_loss'] += diff * 10
        #loss_dict[stage + '_loss'] += loss_dict['stage' + '_image_vq_loss']
        loss_dict[stage + '_rec_image_loss'] = rec_image_loss
        loss_dict[stage + '_loss'] += rec_image_loss
        loss_dict[stage + '_logits_global_alignment'] = logits_global_alignment_loss
        loss_dict[stage + '_loss'] += logits_global_alignment_loss
        loss_dict[stage + '_logits_clip_acc'] = logits_clip_acc  
        return local_image_embed, loss_dict


    def forward(self, batch):
        image_features  = self.image_encoder(batch)
        return image_features

    def on_train_epoch_start(self) -> None:
        if self.current_epoch > self.stage1_epochs:
            self.sentence_bank.set_temp(self.current_epoch, self.stage2_epochs + self.stage3_epochs)
        if self.current_epoch == 0:
            # Stage 1 starts
            optimizers, lr_schedulers, _  = self.call_optimization(max_epochs=self.stage1_epochs, warmup_epochs=self.warmup_epochs) # Start training from global [long]
            self.trainer.lr_schedulers = lr_schedulers
            self.trainer.optimizers = optimizers
        elif self.current_epoch == self.stage1_epochs: 
            optimizers, lr_schedulers, _  = self.call_optimization(max_epochs=self.stage2_epochs, warmup_epochs=self.warmup_epochs) # Start training from global [long]
            self.trainer.lr_schedulers = lr_schedulers
            self .trainer.optimizers = optimizers
        elif self.current_epoch == self.stage1_epochs + self.stage2_epochs:
            self.semi = False
            optimizers, lr_schedulers, _  = self.call_optimization(max_epochs=self.stage3_epochs, warmup_epochs=self.warmup_epochs) # Start training from global [long]
            #self.configure_optimizers(pretrain=False)
            self.trainer.lr_schedulers = lr_schedulers
            self .trainer.optimizers = optimizers
      
     
    def on_train_epoch_end(self) -> None:
        if self.current_epoch == self.stage1_epochs - 1: 
            if self.global_rank == 0:      
                self.trainer.save_checkpoint(f"{self.ckpt_path}/stage2_start.ckpt")
        if self.current_epoch == self.stage1_epochs + self.stage2_epochs - 1: 
            if self.global_rank == 0:      
                self.trainer.save_checkpoint(f"{self.ckpt_path}/stage3_start.ckpt")

    def call_optimization(self, max_epochs=None, warmup_epochs=None, slow_text_encoder=False):
        optim_conf = self.configure_optimizers(max_epochs=max_epochs, warmup_epochs=warmup_epochs, slow_text_encoder=slow_text_encoder)
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
            elif 'discriminator' in name:
                continue
            else:
                params.append(param)
        return params, excluded_params
        

    def configure_optimizers(self, max_epochs=None, warmup_epochs=None, slow_text_encoder=False):
        if max_epochs is None:
            max_epochs = self.max_epochs
        if warmup_epochs is None:
            warmup_epochs = self.warmup_epochs
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()
        if slow_text_encoder:
            other_params, text_params = self.exclude_from_text_encoder(self.named_parameters(), weight_decay=self.weight_decay)
            params = [{"params": text_params}, {"params": other_params}]
        if self.optim == "lars":
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optim == "adamw":
            optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if slow_text_encoder:  
            optimizer.param_groups[0]['lr'] = 1e-5 # fix the text_encoder lr 
            optimizer.param_groups[1]['lr'] = self.learning_rate
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
                    warmup_start_lr=self.learning_rate_start, eta_min=self.learning_rate_end),
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

        






        