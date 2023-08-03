from transformers import AutoTokenizer, AutoConfig
from transformers.models.bert.modeling_bert import *


# No causal mask for sentence-wise decoder
class CrossModalityBertDecoder(BertModel):
    def __init__(self, config=None):
        if config is None:
            config =  AutoConfig.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        config.num_hidden_layers = 8
        config.is_decoder = True # 
        config.add_cross_attention = True
        super().__init__(config, False) # no pooling layer for sentence-wise decoder
    
    def forward(self, x, y):
        '''
        x: x
        y: image_embed
        '''
        return super().forward(inputs_embeds=x, encoder_hidden_states=y, return_dict=True)['last_hidden_state']