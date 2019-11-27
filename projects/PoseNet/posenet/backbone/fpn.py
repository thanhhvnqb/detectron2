import torch
import torch.nn.functional as F
from torch import nn

import fvcore.nn.weight_init as weight_init

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN
from detectron2.modeling.backbone.resnet import build_resnet_backbone

from .mobilenetv3 import build_mobilenetv3_rw_backbone

class LastLevelP6P7DW(nn.Module):
    """This module is used in FCOS to generate extra layers, P6 and P7 from P5 feature."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num_levels = 2
        self.in_feature = "p5"
        norm = "GN"
        conv_fcn = []
        conv_fcn.append(Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=not norm,\
                            groups=out_channels, norm=get_norm(norm, in_channels), activation=F.relu))
        conv_fcn.append(Conv2d(out_channels, out_channels, kernel_size=1, bias=not norm,\
                            norm=get_norm(norm, out_channels), activation=F.relu))
        self.add_module('p6', nn.Sequential(*conv_fcn))

        conv_fcn = []
        conv_fcn.append(Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=not norm,\
                            groups=out_channels, norm=get_norm(norm, out_channels), activation=F.relu))
        conv_fcn.append(Conv2d(out_channels, out_channels, kernel_size=1, bias=not norm,\
                            norm=get_norm(norm, out_channels), activation=F.relu))
        self.add_module('p7', nn.Sequential(*conv_fcn))

        for layer in [*self.p6, *self.p7]:
            weight_init.c2_msra_fill(layer)

    def forward(self, p5):
        p6 = self.p6(p5)
        p7 = self.p7(p6)
        return [p6, p7]

class LastLevelP6P7(nn.Module):
    """This module is used in FCOS to generate extra layers, P6 and P7 from P5 feature."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num_levels = 2
        self.in_feature = "p5"
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, p5):
        p6 = self.p6(p5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]

@BACKBONE_REGISTRY.register()
def build_fcos_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(out_channels, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_mb3_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_mobilenetv3_rw_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7DW(out_channels, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
