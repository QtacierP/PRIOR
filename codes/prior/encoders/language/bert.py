from requests import models
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, BertConfig, AutoModelForMaskedLM, AutoConfig
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
import pytorch_lightning as pl
import torch.nn as nn
import torch
import os
import sys
sys.path.append(os.getcwd())


class ClinicalBERT(nn.Module):
    def __init__(self, pretrained=True, vocab_size=28996):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        else:
            self.encoder = BertModel(BertConfig(hidden_size=768, vocab_size=vocab_size))
        
    def forward(self, x):
        return self.encoder(**x)
    
    def get_width(self):
        return self.encoder.config.hidden_size