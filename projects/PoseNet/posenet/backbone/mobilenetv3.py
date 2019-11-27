import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import FrozenBatchNorm2d, NaiveSyncBatchNorm
from detectron2.layers.wrappers import BatchNorm2d
from .geffnet.helpers import load_pretrained
from .geffnet.efficientnet_builder import *
from .geffnet.mobilenetv3 import model_urls
from .geffnet.mobilenetv3 import MobileNetV3 as geffnet_MobileNetV3

def get_norm(norm):
    """
    Args:
        norm (str or callable):

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm2d,
            "SyncBN": NaiveSyncBatchNorm,
            "FrozenBN": FrozenBatchNorm2d,
            "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
        }[norm]
    return norm

class MobileNetV3(Backbone):
    """ MobileNet-V3

    A this model utilizes the MobileNet-v3 specific 'efficient head', where global pooling is done before the
    head convolution without a final batch-norm layer before the classifier.

    Paper: https://arxiv.org/abs/1905.02244
    """

    def __init__(self, block_args, out_features, out_feature_channels, out_feature_strides, num_classes=1000, in_chans=3, stem_size=16, num_features=1280, head_bias=True,
                 channel_multiplier=1.0, pad_type='', act_layer=HardSwish, drop_rate=0., drop_connect_rate=0.,
                 se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, weight_init='goog'):
        super(MobileNetV3, self).__init__()
        self.drop_rate = drop_rate
        self._out_features = out_features
        self._out_feature_channels = out_feature_channels
        self._out_feature_strides = out_feature_strides
        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = select_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        norm_layer = get_norm(norm_layer)
        self.bn1 = norm_layer(stem_size)  # Modify for FrozenBN
        self.act1 = act_layer(inplace=True)
        in_chs = stem_size

        builder = EfficientNetBuilder(
            channel_multiplier, pad_type=pad_type, act_layer=act_layer, se_kwargs=se_kwargs,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, drop_connect_rate=drop_connect_rate)
        self.blocks = nn.Sequential(*builder(in_chs, block_args))
        in_chs = builder.in_chs
        if "linear" in out_features:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.conv_head = select_conv2d(in_chs, num_features, 1, padding=pad_type, bias=head_bias)
            self.act2 = act_layer(inplace=True)
            self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if weight_init == 'goog':
                initialize_weight_goog(m)
            else:
                initialize_weight_default(m)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([
            self.global_pool, self.conv_head, self.act2,
            nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)
    
    def forward(self, x):
        outputs = {}
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        out_features = self._out_features
        if "stem" in out_features:
            outputs["stem"] = x
        for bidx, block in enumerate(self.blocks):
            if bidx < 6 or "linear" in out_features:
                x = block(x)
            if bidx == 0 and "res1" in out_features:
                outputs["res1"] = x
            elif bidx == 1 and "res2" in out_features:
                outputs["res2"] = x
            elif bidx == 2 and "res3" in out_features:
                outputs["res3"] = x
            elif bidx == 4 and "res4" in out_features:
                outputs["res4"] = x
            elif bidx == 5 and "res5" in out_features:
                outputs["res5"] = x
                
        if "linear" in out_features:
            x = self.global_pool(x)
            x = self.conv_head(x)
            x = self.act2(x)
            x = x.flatten(1)
            if self.drop_rate > 0.:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            outputs["linear"] = self.classifier(x)
        return outputs


def _create_model(model_kwargs, variant, pretrained=False):
    as_sequential = model_kwargs.pop('as_sequential', False)
    model = MobileNetV3(**model_kwargs)
    if pretrained and model_urls[variant]:
        load_pretrained(model, model_urls[variant])
    if as_sequential:
        model = model.as_sequential()
    return model


def _gen_mobilenet_v3_rw(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MobileNet-V3 model (RW variant).

    Paper: https://arxiv.org/abs/1905.02244

    This was my first attempt at reproducing the MobileNet-V3 from paper alone. It came close to the
    eventual Tensorflow reference impl but has a few differences:
    1. This model has no bias on the head convolution
    2. This model forces no residual (noskip) on the first DWS block, this is different than MnasNet
    3. This model always uses ReLU for the SE activation layer, other models in the family inherit their act layer
       from their parent block
    4. This model does not enforce divisible by 8 limitation on the SE reduction channel count

    Overall the changes are fairly minor and result in a very small parameter count difference and no
    top-1/5

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_nre_noskip'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c960'],  # hard-swish
    ]
    out_feature_channels = {f:{"res2": 24, "res3": 40, "res4": 112, "res5": 160}[f] for f in kwargs['out_features']}
    kwargs['out_feature_channels'] = {k:round_channels(c, channel_multiplier) for k, c in out_feature_channels.items()}
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        head_bias=False,  # one of my mistakes
        channel_multiplier=channel_multiplier,
        act_layer=resolve_act_layer(kwargs, 'hard_swish'),
        se_kwargs=dict(gate_fn=get_act_fn('hard_sigmoid'), reduce_mid=True),
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs,
    )
    model = _create_model(model_kwargs, variant, pretrained)
    return model


@BACKBONE_REGISTRY.register()
def build_mobilenetv3_rw_backbone(cfg, input_shape):
    """ MobileNet-V3 RW
    Attn: See note in gen function for this variant.
    """
    # NOTE for train set drop_rate=0.2
    # pretrained model trained with non-default BN epsilon
    kwargs = dict()
    # kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['norm_layer'] = cfg.MODEL.MOBILENETV3.NORM
    out_features = cfg.MODEL.MOBILENETV3.OUT_FEATURES
    kwargs['out_features'] = out_features
    kwargs['out_feature_strides'] = {f:{"res2": 2**2, "res3": 2**3, "res4": 2**4, "res5": 2**5}[f] for f in out_features}
    channel_multiplier = cfg.MODEL.MOBILENETV3.CHANNEL_MULTIPLIER
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    model = _gen_mobilenet_v3_rw('mobilenetv3_rw', channel_multiplier=channel_multiplier, **kwargs)
    if freeze_at > 1:
        for param in [*model.conv_stem.parameters(), *model.bn1.parameters(), *model.act1.parameters()]: 
            param.requires_grad = False
        model.conv_stem = FrozenBatchNorm2d.convert_frozen_batchnorm(model.conv_stem)
        model.bn1 = FrozenBatchNorm2d.convert_frozen_batchnorm(model.bn1)
        model.act1 = FrozenBatchNorm2d.convert_frozen_batchnorm(model.act1)
    if freeze_at >= 2:
        for bidx in range(2, freeze_at + 1):
            for param in model.blocks[bidx - 2].parameters(): 
                param.requires_grad = False
            model.blocks[bidx - 2] = FrozenBatchNorm2d.convert_frozen_batchnorm(model.blocks[bidx - 2])
    return model
