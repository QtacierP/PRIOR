from curses import qiflush
from multiprocessing.pool import Pool
from re import sub
import torch.nn as nn
import torch
import os
import sys
sys.path.append(os.getcwd())
from transformers import AutoTokenizer
import logging 
log = logging.getLogger(__name__)


class AttentionPool(nn.Module):
    """ Self attention pooling Layer"""
    def __init__(self, in_dim):
        super(AttentionPool, self).__init__()
        self.chanel_in = in_dim
        self.query = nn.Linear(in_dim, in_dim//8)
        self.key =  nn.Linear(in_dim, in_dim//8)
        self.softmax  = nn.Softmax(dim=0) #
        
    def forward(self,x):
        """
            inputs :s
                x : input feature in sentence (N, D)
            returns :
                out : self attention value + input feature 
                attention: N .
        """
        n, d  = x.size()
        proj_query  = self.query(x)  # N x D  -> N x M
        proj_key =  self.key(x)  #  N x D  -> N x M
        energy =  proj_query @ proj_key.T # N x M @ M x N = N x N
        attention = self.softmax(energy.sum(1, keepdim=True)) # N x 1 
        return attention.T @ x

class SentenceGather(nn.Module):
    # Word-level gather
    def __init__(self, pool='avg', embed_dim=768):
        super().__init__()
        if pool == 'avg':
            self.pool = lambda x: torch.mean(x, dim=0, keepdim=True)
        elif pool == 'max':
            self.pool = lambda x: torch.max(x, dim=0, keepdim=True)
        elif pool == 'min':
            self.pool = lambda x: torch.min(x, dim=0, keepdim=True)
        elif pool == 'attention':
            self.attention_pool = AttentionPool(embed_dim)
            self.pool = lambda x: self.attention_pool(x)
        else:
            raise NotImplementedError
        log.info('LocalGather: {}'.format(pool))
    

    def forward(self, x, batch):
        roi_mask = batch['text_meta']['sentence_index']
        roi_words = []
        for b in range(x.shape[0]):
            roi_mask_per_text = roi_mask[b]
            text = x[b]
            roi_words_per_text = []
            for i in range(torch.max(roi_mask_per_text).item()):
                aggregate_word = text[torch.where(roi_mask_per_text == (i  + 1))[0], :]
                if aggregate_word.size(0) == 0:
                    continue
                roi_words_per_text.append(self.pool(aggregate_word))
            roi_words_per_text = torch.cat(roi_words_per_text, dim=0)
            roi_words.append(roi_words_per_text)
        return roi_words

