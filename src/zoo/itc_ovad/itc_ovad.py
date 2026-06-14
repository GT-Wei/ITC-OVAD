"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['ITCOVAD', ]


@register
class ITCOVAD(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None, class_text=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
        if isinstance(class_text, torch.Tensor):
            x = self.backbone.forward_image(x)
            class_embeddings = class_text
        else:
            x, class_embeddings = self.backbone(x, class_text)
        x, v_aug_class_embeddings = self.encoder(x, class_embeddings) 
        x = self.decoder(x, class_embeddings, v_aug_class_embeddings, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
