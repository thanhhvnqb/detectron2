# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_posenet_config(cfg):
    """
    Add config for PoseNet.
    """
    _C = cfg
