# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import math
from typing import List
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from lamidetr.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from lamidetr.utils import (inverse_sigmoid, is_dist_avail_and_initialized,
                          load_class_freq, get_fed_loss_inds, get_rank)

from .postprocessing import detector_postprocess
from lamidetr.structures import Boxes, ImageList, Instances

import open_clip


class LaMI_INFER(nn.Module):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2203.03605>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(
        self,
        backbone: nn.Module,
        position_embedding: nn.Module,
        neck: nn.Module,
        transformer: nn.Module,
        text_backbone: nn.Module,
        embed_dim: int,
        classifier,
        pad_len=200,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [58.395, 57.120, 57.375],
        select_box_nums_for_evaluation: int = 300,
        device="cuda",
        vlm_temperature: float =100.0,
        beta: float =0.25,
        score_ensemble: bool = True,
        clip_head_path=None,
        text_backbone_path=None,
    ):
        super().__init__()
        self.vlm_temperature = vlm_temperature
        self.beta = beta
        clip_head = torch.load(clip_head_path)
        self.identical, self.thead = clip_head[0]
        self.head = clip_head[1]

        # define backbone and position embedding module
        self.backbone = backbone
        self.position_embedding = position_embedding

        # define text branch
        state_dict = torch.load(text_backbone_path)
        self.text_backbone = text_backbone
        self.text_backbone.load_state_dict(state_dict, strict=False)
        self.tokenizer = open_clip.get_tokenizer('convnext_large_d_320')

        # define neck module
        self.neck = neck

        # number of dynamic anchor boxes and embedding dimension
        self.embed_dim = embed_dim

        # define transformer module
        self.transformer = transformer

        # define classification head and box head
        self.class_embed = classifier


        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)

        # normalizer for input raw images
        self.device = device
        self.pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        
        # self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # initialize weights
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers + 1
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
        # self.class_linear = nn.ModuleList([copy.deepcopy(self.class_linear) for i in range(num_pred)])
        # self.class_bias = nn.ModuleList([copy.deepcopy(self.class_bias) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

        # two-stage
        # self.transformer.decoder.class_linear = self.class_linear
        # self.transformer.decoder.class_bias = self.class_bias
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # hack implementation for two-stage
        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)

        # set topk boxes selected for inference
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        feat_dim = classifier.zs_weight_dim
        self.content_layer = nn.Linear(feat_dim, embed_dim)

        self.pad_len = pad_len

    def forward(self, images, names, visual_descs, ori_sizes):
        """Forward function of `DINO` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(images)# [1, 3, 800, 1200]
        texts = self.make_description(names, visual_descs)
        content_query_embeddings = []
        for text in texts:
            content_query_embedding = self.preprocess_text(text)
            content_query_embeddings.append(content_query_embedding)
        content_query_embeddings = torch.stack(content_query_embeddings, dim=0)# list -> [6, 768]
        content_class_embeddings = content_query_embeddings.permute(1, 0).contiguous()# [6, 768]->[768, 6]
        content_class_embeddings = F.normalize(content_class_embeddings, p=2, dim=0)
        content_query_embeddings = self.content_layer(content_query_embeddings)# [6, 768]->[6, 256]
        content_query_embeddings = F.normalize(content_query_embeddings, p=2, dim=1)# [6, 256]
 
        batch_size, _, H, W = images.tensor.shape# 1, 3, 800, 1200
        img_masks = images.tensor.new_zeros(batch_size, H, W)

        # original features
        features, features_wonorm = self.backbone(images.tensor)  # output feature dict

        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in DINO
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)

        # feed into transformer
        (
            inter_states,
            init_reference,
            inter_references,
            enc_state,
            enc_reference,  # [0..1]
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            attn_masks=[attn_mask, None],
            content_query_embeds=content_query_embeddings,
            content_class_embeds=content_class_embeddings,
            # content_inds=content_inds, 
        )
        # hack implementation for distributed training
        # inter_states[0] += self.label_enc.weight[0, 0] * 0.0
        inter_states[0] += self.content_layer.weight[0, 0] * 0.0

        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl], classifier=content_class_embeddings)
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_queries]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}

        # prepare two stage output
        interm_coord = enc_reference
        interm_class = self.transformer.decoder.class_embed[-1](enc_state, classifier=content_class_embeddings)
        output["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord}

        box_cls = output["pred_logits"]
        box_pred = output["pred_boxes"]

        # ensemble CLIP score
        roi_features_ori = self.extract_region_feature(features_wonorm, box_pred, 'p3')# [1, 900, 768]
        cls_score = box_cls.sigmoid()# [1, 900, num_class]
        vlm_score = roi_features_ori @ content_class_embeddings * self.vlm_temperature# [1, 900, 768] [768, 6] -> [1, 900, 6]
        vlm_score = vlm_score.softmax(dim=-1)
        # 所有类别都是novel类，统一使用beta参数
        cls_score = cls_score ** (1 - self.beta) * vlm_score ** self.beta
        box_cls = cls_score

        results = self.inference(box_cls, box_pred, images.image_sizes, wo_sigmoid=True)
        processed_results = []
        for results_per_image, ori_size, image_size in zip(
            results, ori_sizes, images.image_sizes
        ): 
            height = ori_size[1]
            width = ori_size[0]
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def extract_region_feature(self, features, bbox, layer_name):
        if layer_name == 'p2':
            h, w = features['p2'].shape[-2:]# 50 75
        elif layer_name == 'p3':
            h, w = features['p3'].shape[-2:]# 50 75

        rpn_boxes = box_cxcywh_to_xyxy(bbox)
        rpn_boxes = torch.clamp(rpn_boxes, min=0, max=1)
        for i in range(len(rpn_boxes)):
            rpn_boxes[i][:,[0,2]] = rpn_boxes[i][:,[0,2]] * w
            rpn_boxes[i][:,[1,3]] = rpn_boxes[i][:,[1,3]] * h
        rpn_boxes = [rpn_box for rpn_box in rpn_boxes]
       
        bs = len(rpn_boxes)
        roi_features = torchvision.ops.roi_align(
            # hid,# [2, 768, 50, 66]
            features['p2'] if layer_name == 'p2' else features['p3'],
            rpn_boxes,
            output_size=(15, 15),
            spatial_scale=1.0,
            aligned=True)  # (bs * num_queries, c, 14, 14) [1800, 768, 30, 30]

        if layer_name == 'p2':
            roi_features = self.backbone.downsample_layers[3](roi_features)# [33, 768, 30, 30]->[33, 1536, 15, 15] 
            roi_features = self.backbone.stages[3](roi_features)# [33, 1536, 15, 15]->[33, 1536, 15, 15]
        roi_features = self.identical(roi_features)# [900, 1536, 15, 15]
        roi_features = self.thead(roi_features)# [900, 1536]
        roi_features = self.head(roi_features)# [900, 768] TODO:
        roi_features = roi_features.reshape(bs, -1, roi_features.shape[-1])
        roi_features = nn.functional.normalize(roi_features, dim=-1)# [1, 900, 768]
        return roi_features

    def encode_text(self, texts):
        cast_dtype = self.text_backbone.transformer.get_cast_dtype()
        x = self.text_backbone.token_embedding(texts).to(cast_dtype)
        x = x + self.text_backbone.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        x = self.text_backbone.transformer(x, attn_mask=self.text_backbone.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.text_backbone.ln_final(x)
        x = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)] @ self.text_backbone.text_projection
        return F.normalize(x, dim=-1)

    def preprocess_image(self, images):
        image_norm = []
        for x in images:
            image_norm.append((x - self.pixel_mean) / self.pixel_std)
        # images = [self.normalizer(x.to(self.device)) for x in images]
        
        images = ImageList.from_tensors(image_norm)
        return images

    def make_descriptor_sentence(self, descriptor):
        descriptor = descriptor[0].lower() + descriptor[1:]
        if descriptor.startswith('a') or descriptor.startswith('an'):
            return f"which is {descriptor}"
        elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
            return f"which {descriptor}"
        elif descriptor.startswith('used'):
            return f"which is {descriptor}"
        else:
            return f"which has {descriptor}"
    
    def wordify(self, string):
        word = string.replace('_', ' ')
        return word

    def make_description(self, names, visual_descs):
        gpt_descriptions = []
        for i, name in enumerate(names):
            descriptions = visual_descs[name]
            for j, desc in enumerate(descriptions):
                if desc == '': continue
                descriptions[j] = self.wordify(name) + ', ' + self.make_descriptor_sentence(desc)
            if descriptions == ['']:
                descriptions = [name]
            gpt_descriptions.append(descriptions)
        return gpt_descriptions

    @torch.no_grad()
    def preprocess_text(self, texts):
        for i, text in enumerate(texts):
            tokens = self.tokenizer(text).to(self.device)
            text_embeddings = F.normalize(self.encode_text(tokens)).mean(dim=0)
        return text_embeddings

    def inference(self, box_cls, box_pred, image_sizes, wo_sigmoid=False):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # box_cls.shape: 1, 300, 80
        # box_pred.shape: 1, 300, 4
        if wo_sigmoid:
            prob = box_cls
        else:
            prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

