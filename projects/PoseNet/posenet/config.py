# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_posenet_config(cfg):
    """
    Add config for PoseNet.
    """
    _C = cfg

    _C.MODEL.POSENET_RPN = CN()
    _C.MODEL.POSENET_RPN.NUM_CLASSES = 2  # the number of classes including background
    _C.MODEL.POSENET_RPN.FPN_STRIDES = [8, 16, 32, 64, 128]
    _C.MODEL.POSENET_RPN.PRIOR_PROB = 0.01
    _C.MODEL.POSENET_RPN.INFERENCE_TH = 0.
    _C.MODEL.POSENET_RPN.NMS_TH = 0.6
    _C.MODEL.POSENET_RPN.PRE_NMS_TOP_N = 1500
    _C.MODEL.POSENET_RPN.POST_NMS_TOP_N = 100

    # Focal loss parameter: alpha
    _C.MODEL.POSENET_RPN.LOSS_ALPHA = 0.25
    # Focal loss parameter: gamma
    _C.MODEL.POSENET_RPN.LOSS_GAMMA = 2.0
    # the number of convolutions used in the cls and bbox tower
    _C.MODEL.POSENET_RPN.NUM_CONVS = 4
    # normalizing the regression targets with FPN strides
    _C.MODEL.POSENET_RPN.NORM_REG_TARGETS = True
    # using center sampling and GIoU.
    # Please refer to https://github.com/yqyao/FCOS_PLUS
    _C.MODEL.POSENET_RPN.CENTER_SAMPLING_RADIUS = 1.5
    _C.MODEL.POSENET_RPN.IOU_LOSS_TYPE = "giou"

    _C.MODEL.POSENET_KPFC = CN()
    _C.MODEL.POSENET_KPFC.NUM_FC = 2
    _C.MODEL.POSENET_KPFC.FC_DIM = 1024

    _C.TEST.BBOX_AUG = CN()
    # Enable test-time augmentation for bounding box detection if True
    _C.TEST.BBOX_AUG.ENABLED = False
