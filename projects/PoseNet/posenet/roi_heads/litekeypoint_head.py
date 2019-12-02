# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.nn import functional as F

import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, interpolate, get_norm
from detectron2.structures import heatmaps_to_keypoints
from detectron2.utils.events import get_event_storage
from detectron2.modeling import ROI_KEYPOINT_HEAD_REGISTRY

_TOTAL_SKIPPED = 0

def build_keypoint_head(cfg, input_shape):
    """
    Build a keypoint head from `cfg.MODEL.ROI_KEYPOINT_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_KEYPOINT_HEAD.NAME
    return ROI_KEYPOINT_HEAD_REGISTRY.get(name)(cfg, input_shape)


def keypoint_rcnn_loss(pred_keypoint_logits, instances, normalizer):
    """
    Arguments:
        pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
            of instances in the batch, K is the number of keypoints, and S is the side length
            of the keypoint heatmap. The values are spatial logits.
        instances (list[Instances]): A list of M Instances, where M is the batch size.
            These instances are predictions from the model
            that are in 1:1 correspondence with pred_keypoint_logits.
            Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
            instance.
        normalizer (float): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.

    Returns a scalar tensor containing the loss.
    """
    heatmaps = []
    valid = []

    keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        keypoints = instances_per_image.gt_keypoints
        heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
            instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return pred_keypoint_logits.sum() * 0

    N, K, H, W = pred_keypoint_logits.shape
    pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(
        pred_keypoint_logits[valid], keypoint_targets[valid], reduction="sum"
    )

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.numel()
    keypoint_loss /= normalizer

    return keypoint_loss


def keypoint_rcnn_inference(pred_keypoint_logits, pred_instances):
    """
    Post process each predicted keypoint heatmap in `pred_keypoint_logits` into (x, y, score, prob)
        and add it to the `pred_instances` as a `pred_keypoints` field.

    Args:
        pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
           of instances in the batch, K is the number of keypoints, and S is the side length of
           the keypoint heatmap. The values are spatial logits.
        pred_instances (list[Instances]): A list of M Instances, where M is the batch size.

    Returns:
        None. boxes will contain an extra "pred_keypoints" field.
            The field is a tensor of shape (#instance, K, 3) where the last
            dimension corresponds to (x, y, probability).
    """
    # flatten all bboxes from all images together (list[Boxes] -> Nx4 tensor)
    bboxes_flat = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)

    keypoint_results = heatmaps_to_keypoints(pred_keypoint_logits.detach(), bboxes_flat.detach())
    num_instances_per_image = [len(i) for i in pred_instances]
    keypoint_results = keypoint_results.split(num_instances_per_image, dim=0)

    for keypoint_results_per_image, instances_per_image in zip(keypoint_results, pred_instances):
        # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y, score, prob)
        keypoint_xyp = keypoint_results_per_image[:, :, [0, 1, 3]]
        instances_per_image.pred_keypoints = keypoint_xyp


@ROI_KEYPOINT_HEAD_REGISTRY.register()
class LitePoseDWConvDeconvUpsampleHead(nn.Module):
    """
    A standard keypoint head containing a series of 3x3 convs, followed by
    a transpose convolution and bilinear interpolation for upsampling.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
            num_keypoints: number of keypoint heatmaps to predicts, determines the number of
                           channels in the final output.
        """
        super().__init__()

        # fmt: off
        # default up_scale to 2 (this can eventually be moved to config)
        up_scale      = 2
        conv_dims     = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS
        num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        expand_factor = cfg.MODEL.ROI_KEYPOINT_HEAD.DWEXPAND_FACTOR
        norm          = cfg.MODEL.ROI_KEYPOINT_HEAD.NORM
        in_channels   = input_shape.channels
        # fmt: on
        norm = ""
        conv_fcn = []
        for layer_channel in conv_dims:
            conv_fcn.append(Conv2d(in_channels, layer_channel, kernel_size=1, bias=not norm,\
                               norm=get_norm(norm, layer_channel), activation=F.relu))
            conv_fcn.append(Conv2d(layer_channel, layer_channel, kernel_size=3, padding=1, bias=not norm,\
                               groups=layer_channel, norm=get_norm(norm, layer_channel), activation=F.relu))
            in_channels = layer_channel
        self.add_module('conv_fcn', nn.Sequential(*conv_fcn))

        deconv_kernel = 4
        self.score_lowres = ConvTranspose2d(
            in_channels, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        self.up_scale = up_scale
        for layer in [*self.conv_fcn, self.score_lowres]:
            weight_init.c2_msra_fill(layer)
        # for name, param in self.named_parameters():
        #     if "bias" in name:
        #         nn.init.constant_(param, 0)
        #     elif "weight" in name:
        #         print("init:", name)
        #         if ".norm" in name:
        #             weight_init.c2_msra_fill(param)
        #         else:
        #             # Caffe2 implementation uses MSRAFill, which in fact
        #             # corresponds to kaiming_normal_ in PyTorch
        #             nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv_fcn(x)
        x = self.score_lowres(x)
        x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x

class EDWConv(nn.Module):
    def __init__(self, in_channels, expand_channel, layer_channel, norm=""):
        super().__init__()
        self.edwconv = Conv2d(in_channels, expand_channel, kernel_size=3, padding=1, bias=not norm,\
                               groups=in_channels, norm=get_norm(norm, expand_channel), activation=F.relu)
        self.reconv = Conv2d(expand_channel, layer_channel, kernel_size=1, bias=not norm,\
                               norm=get_norm(norm, layer_channel), activation=None)
        for layer in [self.edwconv, self.reconv]:
                weight_init.c2_msra_fill(layer)
            
    def forward(self, x):
        out = self.edwconv(x)
        out = self.reconv(out)
        return out + x

@ROI_KEYPOINT_HEAD_REGISTRY.register()
class LitePoseEDWConvDeconvUpsampleHead(nn.Module):
    """
    A standard keypoint head containing a series of 3x3 convs, followed by
    a transpose convolution and bilinear interpolation for upsampling.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
            num_keypoints: number of keypoint heatmaps to predicts, determines the number of
                           channels in the final output.
        """
        super().__init__()

        # fmt: off
        # default up_scale to 2 (this can eventually be moved to config)
        up_scale        = 2
        conv_dims       = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS
        num_keypoints   = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        dwexpand_factor = cfg.MODEL.ROI_KEYPOINT_HEAD.DWEXPAND_FACTOR
        norm            = cfg.MODEL.ROI_KEYPOINT_HEAD.NORM
        in_channels     = input_shape.channels
        # fmt: on
        norm = ""
        self.conv_fcns = []
        for idx, layer_channel in enumerate(conv_dims):
            expand_channel = in_channels * dwexpand_factor
            conv_fcn = EDWConv(in_channels, expand_channel, layer_channel, norm)
            in_channels = layer_channel
            self.add_module("conv_fcn{}".format(idx + 1), conv_fcn)
            self.conv_fcns.append(conv_fcn)

        deconv_kernel = 4
        self.score_lowres = ConvTranspose2d(
            in_channels, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        self.up_scale = up_scale
        for layer in [self.score_lowres]:
            weight_init.c2_msra_fill(layer)
        # for name, param in self.named_parameters():
        #     if "bias" in name:
        #         nn.init.constant_(param, 0)
        #     elif "weight" in name:
        #         print("init:", name)
        #         if ".norm" in name:
        #             weight_init.c2_msra_fill(param)
        #         else:
        #             # Caffe2 implementation uses MSRAFill, which in fact
        #             # corresponds to kaiming_normal_ in PyTorch
        #             nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        for block in self.conv_fcns:
            x = block(x)
            x = F.relu(x)
        # x = self.conv_fcn(x)
        x = self.score_lowres(x)
        x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x
