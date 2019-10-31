# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_posenet_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg

    _C.MODEL.POSENET = CN()

    # Number of branches for TridentNet.
    _C.MODEL.POSENET.NUM_BRANCH = 3
    # normalizing the regression targets with FPN strides
    _C.MODEL.POSENET.NORM_REG_TARGETS: True
    # using center sampling and GIoU.
    # Please refer to https://github.com/yqyao/FCOS_PLUS
    _C.MODEL.POSENET.CENTER_SAMPLING_RADIUS: 1.5
    _C.MODEL.POSENET.IOU_LOSS_TYPE: "giou"
