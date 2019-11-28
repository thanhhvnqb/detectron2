# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import ROI_BOX_HEAD_REGISTRY


@ROI_BOX_HEAD_REGISTRY.register()
class LiteFastRCNNConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        dim_expand_factor = cfg.MODEL.ROI_BOX_HEAD.DWEXPAND_FACTOR
        expand_channels = dim_expand_factor * conv_dim
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        in_channels = input_shape.channels
        if num_conv > 0:
            conv = []
            for k in range(num_conv):
                conv.append(Conv2d(in_channels, expand_channels, kernel_size=1, bias=not norm,\
                                norm=get_norm(norm, expand_channels), activation=F.relu))
                conv.append(Conv2d(expand_channels, expand_channels, kernel_size=5, padding=2, bias=not norm,\
                                groups=expand_channels, norm=get_norm(norm, expand_channels), activation=F.relu))
                in_channels = expand_channels
            conv.append(Conv2d(expand_channels, conv_dim, kernel_size=1, bias=not norm,\
                                norm=get_norm(norm, conv_dim), activation=F.relu))

            self.add_module('conv', nn.Sequential(*conv))
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])
        else:
            self.conv = None

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        if self.conv is not None:
            for layer in self.conv:
                weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        x = self.conv(x) if self.conv is not None else x
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_size(self):
        return self._output_size


def build_box_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)
