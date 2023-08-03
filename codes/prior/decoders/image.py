import torch.nn as nn
import torch
import torch.nn.functional as F
import segmentation_models_pytorch.base as md
from kornia.losses import ssim

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            output_size,
            connection_dropout=0.0,
            use_batchnorm=True,
            attention_type=None
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels)
        self.output_size = output_size
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)
        self.connection_dropout_rate = connection_dropout
        self.dropout = nn.Dropout2d(connection_dropout)

    def forward(self, x, skip=None):
        if skip is not None:
            # print(x.shape, skip.shape)
            if self.connection_dropout_rate > 0:
                skip = self.dropout(skip)
            x = torch.cat([x, skip], dim=1)
        x = F.interpolate(x, self.output_size, mode="nearest")
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

def get_features_size(encoder_name):
    if encoder_name == 'resnet50':
        return [14, 28, 56, 112, 224], [2048, 1024, 512, 256, 64]
    else:
        raise NotImplementedError

def clear(self, *args):
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]

    def _clear(f):
        if not hasattr(self, f):
            return
        attr = getattr(self, f)
        if isinstance(attr, torch.Tensor):
            attr.set_()
        elif isinstance(attr, list):
            del attr[:]
        else:
            setattr(self, f, None)
    for key in args:
        _clear(key)
    return self

class SpatialDropout(nn.Module):
    def __init__(self, p=0.5, is_train=True):
        super(SpatialDropout, self).__init__()
        self.p = p
        self.noise = torch.Tensor()
        self.is_train = is_train

    def forward(self, input):
        self.output.resize_as_(input).copy_(input)
        if self.is_train:
            if input.dim() == 4:
                self.noise.resize_(input.size(0), input.size(1), 1, 1)
            else:
                raise RuntimeError('Input must be 4D (nbatch, nfeat, h, w)')

            self.noise.bernoulli_(1 - self.p)
            # We expand the random dropouts to the entire feature map because the
            # features are likely correlated across the map and so the dropout
            # should also be correlated.
            self.output.mul_(self.noise.expand_as(input))
        else:
            self.output.mul_(1 - self.p)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.is_train:
            self.gradInput.resize_as_(gradOutput).copy_(gradOutput)
            self.gradInput.mul_(self.noise.expand_as(input))  # simply mask the gradients with the noise vector
        else:
            raise RuntimeError('backprop only defined while training')

        return self.gradInput

    def setp(self, p):
        self.p = p

    def __repr__(self):
        return super(SpatialDropout, self).__repr__()

    def clearState(self):
        clear(self, 'noise')
        return super(SpatialDropout, self).clearState()

class ImageDecoder(nn.Module):
    def __init__(self, embed_dim, output_size=128, encoder_name='resnet50', num_colors=1, image_dropout=0.5, dropout_mode='channel', **kwargs) -> None:
        super().__init__()
        in_channels = []
        out_channels = []
        out_size = []
        size_list, channel_list = get_features_size(encoder_name)
        n_blocks = len(channel_list)
        in_ch = embed_dim 
        if dropout_mode == 'channel':
            self.image_dropout = nn.Dropout2d(image_dropout)
        elif dropout_mode == 'spatial':
            self.image_dropout = SpatialDropout(image_dropout)
        for i in range(n_blocks):
            in_channels.append(in_ch)
            out_channels.append(in_ch // 2)
            out_size.append(size_list[i])
            in_ch //= 2
        self.output_head = nn.Sequential(nn.Conv2d(in_ch, num_colors, kernel_size=3, padding=1))
        blocks = [
            DecoderBlock(in_ch, out_ch, size, **kwargs)
            for in_ch, out_ch, size in zip(in_channels, out_channels, out_size)
        ]
        self.blocks = nn.ModuleList(blocks)


    def forward(self, image_embed, text_embed, image):
        # mask image features
        image_embed = self.image_dropout(image_embed)
        # fuse image and text features
        z = torch.cat([image_embed, text_embed], dim=1) 
        for i, block in enumerate(self.blocks):
            z = block(z)
        output = self.output_head(z)
        loss = self.loss_fn(output, image)
        return {'loss': loss, 'output': output}
    
    def loss_fn(self, recon_image, image):
        # TODO: Deep supervision loss may be added ?
        if image.size(2) != recon_image.size(2) or image.size(3) != recon_image.size(3):
            image = F.interpolate(image, size=recon_image.size()[2:], mode='bilinear', align_corners=True)
        recon_loss = F.l1_loss(recon_image, image)
        return recon_loss
    


   