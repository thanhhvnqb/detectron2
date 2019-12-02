# -*- coding: utf-8 -*-

from detectron2.config import CfgNode as CN


def add_posenet_config(cfg):
    """
    Add config for PoseNet.
    """
    _C = cfg
    _C.MODEL.MOBILENETV3 = CN()
    _C.MODEL.MOBILENETV3.NORM = "FrozenBN"
    _C.MODEL.MOBILENETV3.CHANNEL_MULTIPLIER = 1.0
    _C.MODEL.MOBILENETV3.OUT_FEATURES = ["res3", "res4", "res5"]

    ## Add config to exist group
    _C.MODEL.RPN.DWEXPAND_FACTOR = 6
    _C.MODEL.RPN.NORM = "GN"
    _C.MODEL.ROI_BOX_HEAD.DWEXPAND_FACTOR = 1
    _C.MODEL.ROI_BOX_HEAD.NORM = "GN"
    _C.MODEL.ROI_KEYPOINT_HEAD.NORM = "GN"
    _C.MODEL.ROI_KEYPOINT_HEAD.DWEXPAND_FACTOR = 6

    _C.SOLVER.GRAD_CLIP = 10