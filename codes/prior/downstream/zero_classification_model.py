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
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, cohen_kappa_score, accuracy_score

class ZeroShotClassificationModel(PretrainModel):
    task='zero-shot-classification'
    def __init__(self, text_encoder, image_encoder, gpus,  max_epochs=20, warmup_epochs=2, batch_size=16, temperature=0.1, gahter_pool='avg', local_temperature=0.1, embed_dim=512, test_dataset=None, num_workers=0, pre_trained_path='./'):
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
        
        # Define NLP gather
        self.item_gather = SentenceGather(gahter_pool, embed_dim)

        # cache for loss
        self.last_local_batch_size = None
        self.global_alignment_labels = None
        
        # Define dataset 
        self.test_dataset = test_dataset
        self.num_workers = num_workers
        #self.train_iters_per_epoch = len(self.train_dataset)  // ( len(gpus) * batch_size)

        # for dis-training, log...
        self.gpus = gpus
        self.batch_size = 1
        self.load_pretraind_weights(pre_trained_path)
        
        self.prompt = test_dataset.prompt_tensor




    def load_pretraind_weights(self, pre_trained_path):
        states_dict = torch.load(pre_trained_path, map_location='cpu')
        self.load_state_dict(states_dict, strict=False)



    def test_dataloader(self) :
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)


    def get_global_text_representation(self, local_text_embed_stacks):
        batch_stacks = []
        for local_text_embed in local_text_embed_stacks:
            batch_stacks.append(self.global_text_attention(local_text_embed.unsqueeze(dim=0)))
        return torch.cat(batch_stacks, dim=0)

    def on_test_start(self):
        self.global_prompt_tensor = []
        self.local_prompt_tensor = []
        for task_indx, task_prompt_list in enumerate(self.prompt):
            self.global_prompt_tensor.append([])
            self.local_prompt_tensor.append([])
            for task_prompt in task_prompt_list:
                x = self.text_encoder(task_prompt.to(self.device))
                local_prompt_embeddings =  x['last_hidden_state'][:, 1 :-1, :] # [B, L -2, D] 
                atten_mask = task_prompt['attention_mask'][:, 1:-1] # [B, L -2]
                local_prompt_embeddings = local_prompt_embeddings[atten_mask == 1] # [B * L -2, D]
                local_prompt_embeddings = local_prompt_embeddings.unsqueeze(dim=0) # [1, B * L -2, D]
                local_prompt_embeddings = self.item_gather(self.local_text_projection(local_prompt_embeddings))
                local_prompt_embeddings = torch.cat(local_prompt_embeddings, dim=0).unsqueeze(dim=1) # [ B, 1, D]
                q_local_text_embed, diff, embed_ind = self.sentence_bank(local_prompt_embeddings)
                q_local_text_embed = q_local_text_embed.reshape(local_prompt_embeddings.shape)
                global_text_embed = self.get_global_text_representation(q_local_text_embed)
                # Local: [B, L, D] @ [D, E] -> [B,  L, E]; Global: [B, D] @ [D, E] -> [B, E]
                self.local_prompt_tensor[task_indx].append(q_local_text_embed)
                self.global_prompt_tensor[task_indx].append(global_text_embed)

    

    def global_loss(self, image_embed, text_embed, logit_scale):
        # image_embed: [B, D]
        # text_embed: [1, D]
        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        
        logits_per_image = logit_scale * image_embed @ text_embed.t()  # [B, 1]
        logits_per_text = logit_scale * text_embed @ image_embed.t() # [1, B]
        logits = (logits_per_image + logits_per_text.T) / 2
        return logits

        
    def encode_image(self, image):
        local_image_features, global_image_features, image_features_list = self.image_encoder(image, return_features=True)
    
        return self.local_image_projection(local_image_features), self.global_image_projection(global_image_features)


    def local_loss(self, local_image_embed, local_text_embed, keepdim=False):
        # local image embed: [B, 49, D]
        # local text embed: [1, D]
        B, L, D = local_image_embed.shape
        local_image_embed = local_image_embed.reshape(B * L, D)
        local_text_embed = local_text_embed.squeeze(0)
        #print(local_image_embed.shape, local_text_embed.shape)
        text_to_local_image_embed, text_to_local_image_atten, image_to_local_text_embed, image_to_local_text_atten  = self.local_cross_attention(local_image_embed, local_text_embed) # xxx 
        #print(text_to_local_image_atten.shape, image_to_local_text_atten.shape)
        
        text_to_local_image_atten = text_to_local_image_atten.reshape(B, L, 1)
        image_to_local_text_atten = image_to_local_text_atten.reshape(B, L, 1)
        if keepdim:
            return text_to_local_image_atten, image_to_local_text_atten
        return (text_to_local_image_atten.mean() + image_to_local_text_atten.mean()) / 2
    
    


    def shared_step(self, batch):
        image = batch['image']
        label = batch['label']
        # global loss
        total_logits = []
        total_local_logits = []
        local_image_embed, global_image_embed = self.encode_image(image)
        for task_indx, task_prompt_list in enumerate(self.global_prompt_tensor):
            logits = []
            for prompt_indx, task_prompt in enumerate(task_prompt_list):
                logits.append(self.global_loss(global_image_embed, task_prompt, self.logit_scale))
            total_logits.append(torch.max(torch.cat(logits, dim=1), dim=1, keepdim=True)[0])
        
        for task_indx, task_prompt_list in enumerate(self.local_prompt_tensor):
            logits = []
            max_att = - torch.inf
            mean_att = 0
            att_list = []
            for prompt_indx, task_prompt in enumerate(task_prompt_list):
                att = self.local_loss(local_image_embed, task_prompt)
                if att > max_att:
                    max_att = att
                mean_att += att
            mean_att /= len(task_prompt_list)    
            total_local_logits.append(mean_att.unsqueeze(dim=0).unsqueeze(dim=0)) #[1, 1]
        total_local_logits = torch.cat(total_local_logits, dim=1)
        total_local_prob = F.softmax(total_local_logits, dim=1) 
        total_logits = torch.cat(total_logits, dim=1) 
        total_prob = F.softmax(total_logits, dim=1) + total_local_prob * 0.5 
        #total_logits = torch.cat(total_logits, dim=1)
        total_predict = torch.argmax(total_prob, dim=1)
        #total_predict = total_prob.argmax(dim=1, keepdim=True)
        return total_predict, label



    
    def forward(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch)
        #self.log('train_loss', loss, on_step=True, on_epoch=False)        
        return loss


      
    def on_test_epoch_start(self) -> None:
        self.test_preds = None
        self.test_labels = None
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_dix):
        predict, label = self.shared_step(batch)
        if self.test_labels is None:
            self.test_labels = label.detach().cpu().numpy()
            self.test_preds = predict.detach().cpu().numpy()
        else:
            self.test_labels = np.concatenate([self.test_labels, label.detach().cpu().numpy()], axis=0)
            self.test_preds = np.concatenate([self.test_preds, predict.detach().cpu().numpy()], axis=0)
        #self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)     
        #self.log('val_f1_score', f1, on_step=False, on_epoch=True, sync_dist=True)     
        return 
    
    def test_epoch_end(self, outputs) -> None:
       
        acc_score = accuracy_score(self.test_preds, self.test_labels)
        precision = precision_score(self.test_labels, self.test_preds, average='macro')
        recall = recall_score(self.test_labels, self.test_preds, average='macro')
        report = classification_report(self.test_labels, self.test_preds, digits=5)
        kappa = cohen_kappa_score(self.test_labels, self.test_preds)
        f1 = f1_score(self.test_labels, self.test_preds, average='macro')
        #self.log('test_auc', auc_score,  on_epoch=True, prog_bar=True, rank_zero_only=True, sync_dist=False)
        self.log('test_acc', acc_score,  on_epoch=True, prog_bar=True, rank_zero_only=True, sync_dist=False) 
        self.log('test_precision', precision,  on_epoch=True, prog_bar=True, rank_zero_only=True, sync_dist=False)
        self.log('test_recall', recall,  on_epoch=True, prog_bar=True, rank_zero_only=True, sync_dist=False)
        self.log('test_kappa', kappa,  on_epoch=True, prog_bar=True, rank_zero_only=True, sync_dist=False)
        self.log('test_f1_score', f1,  on_epoch=True, prog_bar=True, rank_zero_only=True, sync_dist=False)
        print(report)