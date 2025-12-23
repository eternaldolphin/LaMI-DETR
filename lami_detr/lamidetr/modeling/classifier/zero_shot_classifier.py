# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from lamidetr.config import configurable
from lamidetr.layers import Linear, ShapeSpec

class ZeroShotClassifier(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int = 0,
        zs_weight_path: str = None,
        eval_zs_weight_path: str = None,
        zs_weight_dim: int = 512,
        use_bias: float = 0.0, 
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature

        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.zs_weight_dim = zs_weight_dim
        self.linear = nn.Linear(input_size, zs_weight_dim)
        if zs_weight_path:
            zs_weight = torch.tensor(
                np.load(zs_weight_path), 
                dtype=torch.float32).permute(1, 0).contiguous() # D x C
            eval_zs_weight = torch.tensor(
                np.load(eval_zs_weight_path), 
                dtype=torch.float32).permute(1, 0).contiguous() # D x C

            if self.norm_weight:
                zs_weight = F.normalize(zs_weight, p=2, dim=0)
                eval_zs_weight = F.normalize(eval_zs_weight, p=2, dim=0)

            self.register_buffer('zs_weight', zs_weight)
            self.register_buffer('eval_zs_weight', eval_zs_weight)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'input_shape': input_shape,
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'zs_weight_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            'zs_weight_dim': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            'use_bias': cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            'norm_weight': cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            'norm_temperature': cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
        }

    def forward(self, x, classifier=None, content_inds=None, additional_class=None):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        x = self.linear(x)
        if classifier is not None:
            zs_weight = classifier
        else:
            if self.training:
                zs_weight = self.zs_weight[:, content_inds] if content_inds is not None else self.zs_weight
                if additional_class is not None:
                    additional_zs_weight = additional_class.t()
                    zs_weight = torch.cat([zs_weight, additional_zs_weight], dim=1)
            else:
                zs_weight = self.eval_zs_weight
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)
        x = torch.matmul(x, zs_weight)
        if self.use_bias:
            x = x + self.cls_bias
        return x
