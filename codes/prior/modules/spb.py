import numpy as np
import torch
from torch import nn, tensor
from torch.autograd import Function, Variable
import torch.nn.functional as F
from math import log

class SPB(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, temp=0.9):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.embed = nn.Embedding(n_embed, dim)
        self.temp = temp
        self.curr_temp = temp

    def set_temp(self, epoch, max_epoch, strategy="fixed"):
        if strategy == "fixed":
            self.curr_temp = self.temp 
        elif strategy == "linear":
            self.curr_temp = self.temp - 0.9 * self.temp * epoch / max_epoch
        elif strategy == "exp":
            self.curr_temp = self.temp * (0.1 ** (epoch / max_epoch))
        
    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = flatten @ self.embed.weight.T
        self.gt_dist = dist
        soft_one_hot = F.gumbel_softmax(dist, tau=self.curr_temp, dim=1, hard=False)
        output = soft_one_hot @ self.embed.weight
        embed_ind = soft_one_hot.argmax(1)
        recon_loss = (output - flatten).abs().mean()
        loss = recon_loss 
        self.dist = dist
        return output, loss, embed_ind

    @ torch.no_grad()
    def query(self, input):
        flatten = input.reshape(-1, self.dim)
        logits = flatten @ self.embed.weight.T
        soft_one_hot = F.gumbel_softmax(logits, tau=self.curr_temp, dim=1, hard=False)
        output = soft_one_hot @ self.embed.weight
        #recon_loss = (output - flatten).abs().mean()
        embed_ind = soft_one_hot.argmax(1)
        return output, embed_ind


    def cal_loss(self, input):
        # calculate kl divergence loss between the qurey distribution and the ground truth distribution
        flatten = input.reshape(-1, self.dim)
        dist = flatten @ self.embed.weight.T
        log_gt_dist = F.log_softmax(self.gt_dist, dim=-1)
        log_dist = F.log_softmax(dist, dim=-1)
        kl_div = F.kl_div(log_dist, log_gt_dist, reduction='batchmean', log_target=True)
        return kl_div
    

   




