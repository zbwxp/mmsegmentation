import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmcv.cnn.bricks import build_norm_layer


class DynHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 norm_cfg,
                 act_cfg,
                 upsample_f,
                 dyn_ch,
                 mask_ch,
                 use_low_level_info=False):
        super(DynHead, self).__init__()

        channels = dyn_ch
        num_bases = 0
        if use_low_level_info:
            num_bases = mask_ch
        num_out_channel = (2 + num_bases) * channels + \
                          channels + \
                          channels * channels + \
                          channels + \
                          channels * num_classes + \
                          num_classes

        self.classifier = nn.Sequential(
            # ASPP(in_channels, aspp_dilate),
            ConvModule(
                in_channels,
                256,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg, ),
            nn.Conv2d(256, num_out_channel, 1)
        )

        nn.init.kaiming_normal_(self.classifier[1].weight)

    def forward(self, feature):
        return self.classifier(feature)


@HEADS.register_module()
class PADFPNHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self,
                 upsample_factor,
                 dyn_branch_ch,
                 mask_head_ch,
                 feature_strides, **kwargs):
        super(PADFPNHead, self).__init__(

            input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        # padhead
        self.upsample_f = upsample_factor
        self.dyn_ch = dyn_branch_ch
        self.mask_ch = mask_head_ch
        self.use_low_level_info = True
        last_stage_ch = self.in_channels[-1]
        self.classifier = DynHead(last_stage_ch,
                                  self.num_classes,
                                  self.norm_cfg,
                                  self.act_cfg,
                                  self.upsample_f,
                                  self.dyn_ch,
                                  self.mask_ch,
                                  self.use_low_level_info)

        self.bottleneck = ConvModule(
            self.in_channels[0],
            self.mask_ch,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        _, norm = build_norm_layer(self.norm_cfg, 2 + self.mask_ch)
        self.add_module("cat_norm", norm)
        nn.init.constant_(self.cat_norm.weight, 1)
        nn.init.constant_(self.cat_norm.bias, 0)
        # # scale heads
        # self.scale_heads = nn.ModuleList()
        # for i in range(len(feature_strides)):
        #     head_length = max(
        #         1,
        #         int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
        #     scale_head = []
        #     for k in range(head_length):
        #         scale_head.append(
        #             ConvModule(
        #                 self.in_channels[i] if k == 0 else self.channels,
        #                 self.channels,
        #                 3,
        #                 padding=1,
        #                 conv_cfg=self.conv_cfg,
        #                 norm_cfg=self.norm_cfg,
        #                 act_cfg=self.act_cfg))
        #         if feature_strides[i] != feature_strides[0]:
        #             scale_head.append(
        #                 nn.Upsample(
        #                     scale_factor=2,
        #                     mode='bilinear',
        #                     align_corners=self.align_corners))
        #     self.scale_heads.append(nn.Sequential(*scale_head))

    def forward(self, inputs):
        features = self._transform_inputs(inputs)
        x = self.classifier(features[-1])

        assert self.use_low_level_info
        x_cat = self.bottleneck(features[0])
        output = self.interpolate(x, x_cat, self.cat_norm)

        # output = self.scale_heads[0](x[0])
        # for i in range(1, len(self.feature_strides)):
        #     # non inplace
        #     output = output + resize(
        #         self.scale_heads[i](x[i]),
        #         size=output.shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        #
        # output = self.cls_seg(output)
        return output


    def interpolate(self, x, x_cat=None, norm=None):
        dy_ch = self.dyn_ch
        B, conv_ch, H, W = x.size()
        x = x.view(B, conv_ch, H * W).permute(0, 2, 1)
        x = x.reshape(B * H * W, conv_ch)
        weights, biases = self.get_subnetworks_params(x, channels=dy_ch)
        f = self.upsample_f
        self.coord_generator(H, W)
        coord = self.coord.reshape(1, H, W, 2, f, f).permute(0, 3, 1, 4, 2, 5).reshape(1, 2, H * f, W * f)
        coord = coord.repeat(B, 1, 1, 1)
        if x_cat is not None:
            coord = torch.cat((coord, x_cat), 1)
            coord = norm(coord)

        B_coord, ch_coord, H_coord, W_coord = coord.size()
        coord = coord.reshape(B_coord, ch_coord, H, f, W, f).permute(0, 2, 4, 1, 3, 5).reshape(1,
                                                                                               B_coord * H * W * ch_coord,
                                                                                               f, f)
        output = self.subnetworks_forward(coord, weights, biases, B * H * W)
        output = output.reshape(B, H, W, self.num_classes, f, f).permute(0, 3, 1, 4, 2, 5)
        output = output.reshape(B, self.num_classes, H * f, W * f)
        return output

    def get_subnetworks_params(self, attns, num_bases=0, channels=16):
        assert attns.dim() == 2
        n_inst = attns.size(0)
        if self.use_low_level_info:
            num_bases = self.mask_ch
        else:
            num_bases = 0

        w0, b0, w1, b1, w2, b2 = torch.split_with_sizes(attns, [
            (2 + num_bases) * channels, channels,
            channels * channels, channels,
            channels * self.num_classes, self.num_classes
        ], dim=1)

        # out_channels x in_channels x 1 x 1
        w0 = w0.reshape(n_inst * channels, 2 + num_bases, 1, 1)
        b0 = b0.reshape(n_inst * channels)
        w1 = w1.reshape(n_inst * channels, channels, 1, 1)
        b1 = b1.reshape(n_inst * channels)
        w2 = w2.reshape(n_inst * self.num_classes, channels, 1, 1)
        b2 = b2.reshape(n_inst * self.num_classes)

        return [w0, w1, w2], [b0, b1, b2]

    def subnetworks_forward(self, inputs, weights, biases, n_subnets):
        assert inputs.dim() == 4
        n_layer = len(weights)
        x = inputs
        # NOTE: x has to be treated as min_batch size 1
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=n_subnets
            )
            if i < n_layer - 1:
                x = F.relu(x)
        return x

    def coord_generator(self, height, width):
        f = self.upsample_f
        coord = compute_locations_per_level(f, f)
        H = height
        W = width
        coord = coord.repeat(H * W, 1, 1, 1)
        self.coord = coord.to(device='cuda')


def compute_locations_per_level(h, w):
    shifts_x = torch.arange(
        0, 1, step=1 / w,
        dtype=torch.float32
    )
    shifts_y = torch.arange(
        0, 1, step=1 / h,
        dtype=torch.float32
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    locations = torch.stack((shift_x, shift_y), dim=0)
    return locations
