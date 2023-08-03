import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class LocalCrossAttention(nn.Module):
    def __init__(self, embed_dim, drop_rate=0):
        super(LocalCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query1 = nn.Linear(embed_dim, embed_dim)
        self.key1 = nn.Linear(embed_dim, embed_dim)
        self.value1 = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(drop_rate)
        self.query2 = nn.Linear(embed_dim, embed_dim)
        self.key2 = nn.Linear(embed_dim, embed_dim)
        self.value2 = nn.Linear(embed_dim, embed_dim)
        self.dropout2 = nn.Dropout(drop_rate)

    def forward(
        self,
        input_tensor1,
        input_tensor2,
        attention_mask1=None,
        attention_mask2=None      
    ):
        # for vision input [B, I]
        query_layer1 = self.query1(input_tensor1)
        key_layer1 = self.key1(input_tensor1)
        value_layer1 = self.value1(input_tensor1)


        # for text input [B, T]
        query_layer2 = self.query2(input_tensor2)
        key_layer2 = self.key2(input_tensor2)
        value_layer2 =  self.value2(input_tensor2)

        
        attention_scores1  = query_layer2 @ key_layer1.T # [T, D] @ [D, I] = [T, I]
        attention_scores1 = attention_scores1 / math.sqrt(self.embed_dim)
        if attention_mask1 is not None:
            attention_scores1 = attention_scores1 + attention_mask1

        # Sigmoid is better in this case
        # TODO: pre-normalize vs. post-normalize 
        attention_probs1 = F.sigmoid(attention_scores1)
     
        attention_probs1 = self.dropout1(attention_probs1)
        context_layer1 =  attention_probs1 @ value_layer1 # [T, I] @ [I, D] = [T, D]
        attention_scores2 = query_layer1 @ key_layer2.T # [I, D] @ [D, T] = [I, T]
        attention_scores2 = attention_scores2 / math.sqrt(self.embed_dim)

        if attention_mask2 is not None:
            attention_scores2 = attention_scores2 + attention_mask2
       
        attention_probs2 = F.sigmoid(attention_scores2)

        attention_probs2 = self.dropout2(attention_probs2)
        context_layer2 = attention_probs2 @ value_layer2 # [I, T] @ [T, D] = [I, D]
        return context_layer2, attention_probs2, context_layer1, attention_probs1