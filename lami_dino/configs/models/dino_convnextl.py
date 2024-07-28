import copy
import torch.nn as nn

from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L

from detrex.modeling.backbone import ConvNeXt
from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from detrex.modeling.classifier import ZeroShotClassifier
from detrex.layers import PositionEmbeddingSine

from lami_dino.modeling import (
    DINO,
    DINOTransformerEncoder,
    DINOTransformerDecoder,
    DINOTransformer,
    DINOCriterion,
)

model = L(DINO)(
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
    transformer=L(DINOTransformer)(
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
        two_stage_num_proposals="${..num_queries}",
    ),
    embed_dim=256,
    num_classes=80,
    num_queries=900,
    aux_loss=True,
    criterion=L(DINOCriterion)(
        num_classes="${..num_classes}",
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 1,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_class_dn": 1,
            "loss_bbox_dn": 5.0,
            "loss_giou_dn": 2.0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
        two_stage_binary_cls=False,
    ),
    classifier=L(ZeroShotClassifier)( 
        input_shape=256,
        num_classes="${..num_classes}", 
        zs_weight_dim=768,
        zs_weight_path="${..query_path}",
        eval_zs_weight_path="${..eval_query_path}",
        norm_weight=True,),
    query_path="dataset/metadata/coco_clip_convnextl_a+cname.npy",
    eval_query_path="dataset/metadata/coco_clip_convnextl_a+cname.npy",
    vlm_query_path = None,
    save_dir = None,
    score_ensemble = False,
    unseen_classes = None,
    seen_classes = None,
    all_classes = None,
    vlm_temperature=100.0,
    alpha=0.3,
    beta=0.7,
    novel_scale=1.0,
    dn_number=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    device="cuda",
    clip_head_path='./pretrained_models/clip_convnext_large_head.pth',
)

# set aux loss weight dict
base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict
