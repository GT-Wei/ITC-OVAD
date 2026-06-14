# Copyright (c) Tencent Inc. All rights reserved.
import itertools
from typing import List, Sequence, Tuple
import torch
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from src.core import register
from src.nn.backbone.CLIP_encoder.ClipImageBackbone import CLIPModifiedResNet
from src.nn.backbone.CLIP_encoder.ClipTextBackbone import TextTransformer
from torch import nn
from src.nn.backbone.presnet import PResNet

__all__ = ['Multi_modality_Backbone']

@register
class Multi_modality_Backbone(nn.Module):
    def __init__(self,
                 image_model,
                 text_model,
                 image_model_name = None,
                 frozen_stages: int = -1,
                 with_text_model: bool = True,
                 init_cfg = None) -> None:
        super(Multi_modality_Backbone, self).__init__()
        self.with_text_model = with_text_model
        
        if image_model_name == 'presnet':
            self.image_model = PResNet(**image_model)
        else:
            self.image_model = CLIPModifiedResNet(**image_model)
        if self.with_text_model:
            self.text_model = TextTransformer(**text_model)
        else:
            self.text_model = None
            
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self.image_model, self.image_model.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super().train(mode)
        self._freeze_stages()

    def forward(self, image: Tensor,
                text: List[List[str]]) -> Tuple[Tuple[Tensor], Tensor]:
        img_feats = self.image_model(image)
        if self.with_text_model:
            txt_feats = self.text_model(text, batch_size=image.shape[0])
            # txt_feats = self.text_feat[:, 0:2, :]
            return img_feats, txt_feats
        else:
            return img_feats, None

    def forward_text(self, text: List[List[str]]) -> Tensor:
        assert self.with_text_model, "forward_text() requires a text model"
        txt_feats = self.text_model(text)
        return txt_feats

    def forward_image(self, image: Tensor) -> Tuple[Tensor]:
        return self.image_model(image)