# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from typing import Dict, List
import logging
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.layers import ShapeSpec
from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY, RPN_HEAD_REGISTRY, build_rpn_head
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from .rpn_outputs import RPNOutputs, find_top_rpn_proposals
from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

logger = logging.getLogger(__name__)


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


@RPN_HEAD_REGISTRY.register()
class StandardPoseRPNHead(nn.Module):
    """
    RPN classification and regression heads. Uses a 3x3 conv to produce a shared
    hidden state from which one 1x1 conv predicts objectness logits for each anchor
    and a second 1x1 conv predicts bounding-box deltas specifying how to deform
    each anchor into an object proposal.
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        num_classes = cfg.MODEL.POSENET_RPN.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.POSENET_RPN.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.POSENET_RPN.NORM_REG_TARGETS

        tower = []
        for _ in range(cfg.MODEL.POSENET_RPN.NUM_CONVS):
            tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True))
            tower.append(nn.GroupNorm(32, in_channels))
            tower.append(nn.ReLU())

        self.add_module('tower', nn.Sequential(*tower) if tower is not None else None)
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        # initialization
        for modules in [self.tower, self.cls_logits, self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    # torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.POSENET_RPN.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            if self.tower is not None:
                tower = self.tower(feature)
            else:
                tower = feature
            logits.append(self.cls_logits(tower))
            centerness.append(self.centerness(tower))

            bbox_pred = self.scales[l](self.bbox_pred(tower))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        return logits, bbox_reg, centerness


@PROPOSAL_GENERATOR_REGISTRY.register()
class PoseRPN(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(PoseRPN, self).__init__()
        self.in_features = cfg.MODEL.RPN.IN_FEATURES
        rpn_head = build_rpn_head(cfg, [input_shape[f] for f in self.in_features])

        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.rpn_head = rpn_head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.POSENET_RPN.FPN_STRIDES
        self.pre_nms_top_n_train = cfg.MODEL.POSENET_RPN.PRE_NMS_TOP_N_TRAIN
        self.post_nms_top_n_train = cfg.MODEL.POSENET_RPN.POST_NMS_TOP_N_TRAIN
        self.pre_nms_top_n_test = cfg.MODEL.POSENET_RPN.PRE_NMS_TOP_N_TEST
        self.post_nms_top_n_test = cfg.MODEL.POSENET_RPN.POST_NMS_TOP_N_TEST

    def forward(self, images, features, gt_instances=None):
        """
        Arguments:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None
        gt_classes = [x.gt_classes + 1 for x in gt_instances] if gt_instances is not None else None
        del gt_instances
        features = [features[f] for f in self.in_features]
        box_cls, box_regression, centerness = self.rpn_head(features)
        locations = self.compute_locations(features)

        if self.training:
            self.box_selector_test.pre_nms_top_n = self.pre_nms_top_n_train
            self.box_selector_test.fpn_post_nms_top_n = self.post_nms_top_n_train
            loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(locations, box_cls, box_regression, centerness, gt_boxes, gt_classes)
            losses = {"loss_rpn_cls": loss_box_cls, "loss_rpn_reg": loss_box_reg, "loss_centerness": loss_centerness}
        else:
            self.box_selector_test.pre_nms_top_n = self.pre_nms_top_n_test
            self.box_selector_test.fpn_post_nms_top_n = self.post_nms_top_n_test
            losses = {}
        boxes = self.box_selector_test(locations, box_cls, box_regression, centerness, images.image_sizes)
        return boxes, losses

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(h, w, self.fpn_strides[level], feature.device)
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


# @PROPOSAL_GENERATOR_REGISTRY.register()
# class PoseRPN(nn.Module):
#     """
#     Region Proposal Network, introduced by the Faster R-CNN paper.
#     """
#     def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
#         super().__init__()

#         # fmt: off
#         self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
#         self.in_features = cfg.MODEL.RPN.IN_FEATURES
#         self.nms_thresh = cfg.MODEL.RPN.NMS_THRESH
#         self.batch_size_per_image = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
#         self.positive_fraction = cfg.MODEL.RPN.POSITIVE_FRACTION
#         self.smooth_l1_beta = cfg.MODEL.RPN.SMOOTH_L1_BETA
#         self.loss_weight = cfg.MODEL.RPN.LOSS_WEIGHT
#         # fmt: on

#         # Map from self.training state to train/test settings
#         self.pre_nms_topk = {
#             True: cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
#             False: cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
#         }
#         self.post_nms_topk = {
#             True: cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
#             False: cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
#         }
#         self.boundary_threshold = cfg.MODEL.RPN.BOUNDARY_THRESH

#         self.anchor_generator = build_anchor_generator(cfg, [input_shape[f] for f in self.in_features])
#         self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
#         self.anchor_matcher = Matcher(cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True)
#         self.rpn_head = build_rpn_head(cfg, [input_shape[f] for f in self.in_features])

#     def forward(self, images, features, gt_instances=None):
#         """
#         Args:
#             images (ImageList): input images of length `N`
#             features (dict[str: Tensor]): input data as a mapping from feature
#                 map name to tensor. Axis 0 represents the number of images `N` in
#                 the input data; axes 1-3 are channels, height, and width, which may
#                 vary between feature maps (e.g., if a feature pyramid is used).
#             gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
#                 Each `Instances` stores ground-truth instances for the corresponding image.

#         Returns:
#             proposals: list[Instances] or None
#             loss: dict[Tensor]
#         """
#         gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None
#         del gt_instances
#         features = [features[f] for f in self.in_features]
#         pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
#         anchors = self.anchor_generator(features)
#         # TODO: The anchors only depend on the feature map shape; there's probably
#         # an opportunity for some optimizations (e.g., caching anchors).
#         outputs = RPNOutputs(
#             self.box2box_transform,
#             self.anchor_matcher,
#             self.batch_size_per_image,
#             self.positive_fraction,
#             images,
#             pred_objectness_logits,
#             pred_anchor_deltas,
#             anchors,
#             self.boundary_threshold,
#             gt_boxes,
#             self.smooth_l1_beta,
#         )

#         if self.training:
#             losses = {k: v * self.loss_weight for k, v in outputs.losses().items()}
#         else:
#             losses = {}

#         with torch.no_grad():
#             # Find the top proposals by applying NMS and removing boxes that
#             # are too small. The proposals are treated as fixed for approximate
#             # joint training with roi heads. This approach ignores the derivative
#             # w.r.t. the proposal boxes’ coordinates that are also network
#             # responses, so is approximate.
#             proposals = find_top_rpn_proposals(
#                 outputs.predict_proposals(),
#                 outputs.predict_objectness_logits(),
#                 images,
#                 self.nms_thresh,
#                 self.pre_nms_topk[self.training],
#                 self.post_nms_topk[self.training],
#                 self.min_box_side_len,
#                 self.training,
#             )
#             # For RPN-only models, the proposals are the final output and we return them in
#             # high-to-low confidence order.
#             # For end-to-end models, the RPN proposals are an intermediate state
#             # and this sorting is actually not needed. But the cost is negligible.
#             inds = [p.objectness_logits.sort(descending=True)[1] for p in proposals]
#             proposals = [p[ind] for p, ind in zip(proposals, inds)]

#         return proposals, losses
