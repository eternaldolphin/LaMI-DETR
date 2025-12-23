import copy
import torch.nn as nn

from lamidetr.layers import ShapeSpec
from lamidetr.config import LazyCall as L

from lamidetr.modeling.backbone import ConvNeXt, TextTransformer
from lamidetr.modeling.neck import ChannelMapper
from lamidetr.modeling.classifier import ZeroShotClassifier
from lamidetr.layers import PositionEmbeddingSine

from lami_detr.modeling import (
    LaMI_INFER,
    DINOTransformerEncoder,
    DINOTransformerDecoder,
    LaMITransformer,
)

model = L(LaMI_INFER)(
    backbone=L(ConvNeXt)(
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        out_indices=(1, 2, 3),
        frozen_stages=4,
        score_ensemble="${..score_ensemble}",
    ),
    text_backbone=L(TextTransformer)(
        context_length=77,
        vocab_size=49408,
        width=768,
        heads=12,
        layers=16,
        ls_init_value=None,
        output_dim=768,
        embed_cls=False,
        pad_id=0,
        output_tokens=False,
        freeze_layer=16,
    ),
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    neck=L(ChannelMapper)(
        input_shapes={
            "p1": ShapeSpec(channels=384),
            "p2": ShapeSpec(channels=768),
            "p3": ShapeSpec(channels=1536),
        },
        in_features=["p1", "p2", "p3"],
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
    ),
    transformer=L(LaMITransformer)(
        encoder=L(DINOTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            post_norm=False,
            num_feature_levels="${..num_feature_levels}",
        ),
        decoder=L(DINOTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels="${..num_feature_levels}",
        ),
        num_feature_levels=4,
    ),
    embed_dim=256,
    classifier=L(ZeroShotClassifier)( 
        input_shape=256,
        zs_weight_dim=768,
        norm_weight=True,),
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    device="cuda",
    score_ensemble = True,
    clip_head_path='../pretrained_models/clip_convnext_large_head.pth',
    text_backbone_path='../pretrained_models/clip_convnext_large_text_backbone.pth'
)

