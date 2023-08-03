
from pyexpat import features
import timm
import torch.nn as nn
from timm.models import resnetv2_50, resnetv2_50d
from torchvision.models import resnet50
from timm.models.layers import StdConv2d
import torch
import torch.nn.functional as F
import sys
import os



class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x[0]


class ResNet(nn.Module):
    def __init__(self, in_channels=1, name=None, pool_method='mean', pretrained=True) -> None:
        super().__init__()
        self.name = name
        if name == 'resnet50':
            self.encoder = resnet50_224(in_channels=in_channels, pretrained=pretrained)
        else:
            raise NotImplementedError
        if pool_method == 'mean':
            self.pool = lambda x: torch.mean(x, dim=[2, 3])
        elif pool_method == 'attention':
            self.atten_pool = AttentionPool2d(spacial_dim=7, embed_dim=2048, num_heads=8)
            self.pool = lambda x: self.atten_pool(x)
        else:
            self.pool = lambda x: x
        self._modify_forward()
        

    def _modify_forward(self):
        if self.name == 'resnet50':
            def forward_wrapper(x, return_features=False):
                features = []
                x = self.encoder.conv1(x)
                x = self.encoder.bn1(x)
                x = self.encoder.relu(x)
                if return_features:
                    features.append(x)
                x = self.encoder.maxpool(x)
                x = self.encoder.layer1(x)
                if return_features:
                    features.append(x)
                x = self.encoder.layer2(x)
                if return_features:
                    features.append(x)
                x = self.encoder.layer3(x)
                if return_features:
                    features.append(x)
                x = self.encoder.layer4(x)
                local_features = x
                if return_features:
                    features.append(x)
                global_x = self.pool(x)
                global_x = global_x.view(global_x.size(0), -1)
                local_features = local_features.view(local_features.shape[0], -1, local_features.shape[1])
                if return_features:
                    return local_features,  global_x, features[::-1]
                return local_features, global_x
        else:
            raise NotImplementedError
        self.encoder.forward = forward_wrapper

        
    def forward(self, x, return_features=False):
        try:
            return self.encoder.forward_features(x)
        except:
            return self.encoder.forward(x, return_features=return_features)
    

    def get_global_width(self):
        try:
            return self.encoder.num_features
        except:
            return 512 * 4

    def get_width(self):
        try:
            return self.encoder.num_features 
        except:
            return 512 * 4 

    def get_local_width(self):
        try:
            return self.encoder.num_features 
        except:
            return 512 * 4 
        
    def get_name(self):
        return self.name
    
    def get_last_spatial_info(self):
        if  self.name == 'resnet50':
            return [7, 7]

          

def resnet50_224(in_channels=3, **kwargs):
    model =  resnet50(**kwargs)
    if in_channels != 3:
        old_conv = model.conv1
        out_channels = model.conv1.out_channels
        model.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3,
                               bias=False)
        model.conv1.weight = torch.nn.Parameter(old_conv.weight.sum(dim=1, keepdim=True))
        model.fc = nn.Identity()
    return model 

if __name__ == '__main__':
    model = ResNet(name='resnet50', in_channels=1, pretrained=True, pool_method='attention')
    x = torch.randn(1, 1, 224, 224)
    _, _, features = model.encoder(x, return_features=True)
    for f in features:
        print(f.shape)