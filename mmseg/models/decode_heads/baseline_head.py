import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPHead, ASPPModule


class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)

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

        nn.init.kaiming_normal_(self.classifier[-1].weight)

    def forward(self, feature):
        return self.classifier(feature)

@HEADS.register_module()
class BaseHead(ASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, c1_in_channels, c1_channels,
                 upsample_factor,
                 dyn_branch_ch,
                 mask_head_ch,
                 pad_out_channel_factor=4,
                 **kwargs):
        super(BaseHead, self).__init__(**kwargs)
        assert c1_in_channels >= 0
        self.pad_out_channel = int(pad_out_channel_factor*c1_channels)
        self.upsample_f = upsample_factor
        self.dyn_ch = dyn_branch_ch
        self.mask_ch = mask_head_ch
        self.use_low_level_info = False
        self.use_dropout = False
        # self.aspp_modules = DepthwiseSeparableASPPModule(
        #     dilations=self.dilations,
        #     in_channels=self.in_channels,
        #     channels=self.channels,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)

        last_stage_ch = self.channels
        # self.classifier = DynHead(last_stage_ch,
        #                           self.pad_out_channel,
        #                           self.norm_cfg,
        #                           self.act_cfg,
        #                           self.upsample_f,
        #                           self.dyn_ch,
        #                           self.mask_ch,)

        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None
        # self.sep_bottleneck = nn.Sequential(
        #     DepthwiseSeparableConvModule(
        #         self.channels + c1_channels,
        #         self.channels,
        #         3,
        #         padding=1,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg),
        #     DepthwiseSeparableConvModule(
        #         self.channels,
        #         self.channels,
        #         3,
        #         padding=1,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg))
        self.sep_bottleneck2 = ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        if not self.use_dropout:
            self.out = nn.Conv2d(self.channels, self.num_classes, 1, padding=0, bias=False)
            nn.init.kaiming_normal_(self.out.weight)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        # aspp_outs = [
        #     resize(
        #         self.image_pool(x),
        #         size=x.size()[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        # ]
        # aspp_outs.extend(self.aspp_modules(x))
        # aspp_outs = torch.cat(aspp_outs, dim=1)
        # output = self.bottleneck(aspp_outs)
        output = x

        # output = self.classifier(output)
        # output = self.interpolate(output)
        plot = False


        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                        output,
                        size=c1_output.size()[2:],
                        mode='bilinear',
                        align_corners=self.align_corners)
            if plot:
                output2 = output
                output3 = c1_output
            output = torch.cat([output, c1_output], dim=1)
        # output = self.sep_bottleneck(output)
        output = self.sep_bottleneck2(output)

        output = resize(
            output,
            size=inputs[0].size()[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.use_dropout:
            output = self.cls_seg(output)
        else:
            output = self.out(output)
        if plot:
            outputs=[]
            outputs.append(output)
            outputs.append(output2)
            outputs.append(output3)
            return outputs

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
        output = output.reshape(B, H, W, self.pad_out_channel, f, f).permute(0, 3, 1, 4, 2, 5)
        output = output.reshape(B, self.pad_out_channel, H * f, W * f)
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
            channels * self.pad_out_channel, self.pad_out_channel
        ], dim=1)

        # out_channels x in_channels x 1 x 1
        w0 = w0.reshape(n_inst * channels, 2 + num_bases, 1, 1)
        b0 = b0.reshape(n_inst * channels)
        w1 = w1.reshape(n_inst * channels, channels, 1, 1)
        b1 = b1.reshape(n_inst * channels)
        w2 = w2.reshape(n_inst * self.pad_out_channel, channels, 1, 1)
        b2 = b2.reshape(n_inst * self.pad_out_channel)

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
