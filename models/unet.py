#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : unet.py
#   Author       : Yufeng Liu
#   Date         : 2021-04-26
#   Description  : 
#
#================================================================

import torch
import torch.nn as nn
import copy
import numpy as np

from neuronet.models.base_model import BaseModel
from neuronet.models.modules.modules import Upsample, ConvDropoutNormNonlin, InitWeights_He, StackedConvLayers

class UpBlock(nn.Module):
    def __init__(self, up_in_channels, in_channels=None, out_channels=None, up_stride=(2,2,2), has_nonlin=True):
        super(UpBlock, self).__init__()
        self.has_nonlin = has_nonlin
        self.up = nn.ConvTranspose3d(up_in_channels, out_channels, up_stride, stride=up_stride)

        if in_channels is None:
            conv_in_channels = out_channels
            self.skip_input = False
        else:
            self.skip_input = True
            conv_in_channels = out_channels + in_channels

        if has_nonlin:
            self.conv = ConvDropoutNormNonlin(conv_in_channels, out_channels)

    def forward(self, x, x_skip=None):
        x = self.up(x)
        if x_skip is not None and self.skip_input:
            x = torch.cat((x, x_skip), dim=1)
        if self.has_nonlin:
            x = self.conv(x)

        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_kernel=(3,3,3), down_stride=(2,2,2)):
        super(DownBlock, self).__init__()
        padding = tuple((k-1)//2 for k in down_kernel)
        down_kwargs = {
            'kernel_size': down_kernel,
            'stride': down_stride,
            'padding': padding,
            'dilation': 1,
            'bias': True,
        }
        self.down = ConvDropoutNormNonlin(in_channels, out_channels, conv_kwargs=down_kwargs)
        conv_kwargs = copy.deepcopy(down_kwargs)
        conv_kwargs['stride'] = 1
        self.conv = ConvDropoutNormNonlin(out_channels, out_channels, conv_kwargs=conv_kwargs)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)

        return x

class UNet(BaseModel):
    MAX_NUM_FILTERS_3D = 320

    def __init__(self, in_channels, base_num_filters, class_num, down_kernel_list, stride_list, num_side_loss, output_bias=False, direct_supervision=True):
        super(UNet, self).__init__()
        assert len(down_kernel_list) == len(stride_list)
        self.downs = []
        self.num_side_loss = num_side_loss
        # if direct_supervision turned on, no nonlinear or reprojection 
        # for the highest resolution image, and the side loss must
        # add additional nonlinear reprojection.
        self.direct_supervision = direct_supervision

        # the first layer to process the input image
        self.pre_layer = nn.Sequential(
            ConvDropoutNormNonlin(in_channels, base_num_filters),
            ConvDropoutNormNonlin(base_num_filters, base_num_filters),
        )

        in_channels = base_num_filters
        out_channels = 2 * base_num_filters
        down_filters = []
        for i in range(len(down_kernel_list)):
            down_kernel = down_kernel_list[i]
            stride = stride_list[i]
            down_filters.append((in_channels, out_channels))
            down = DownBlock(in_channels, out_channels, down_kernel=down_kernel, down_stride=stride)
            self.downs.append(down)
            in_channels = min(out_channels, self.MAX_NUM_FILTERS_3D)
            out_channels = min(out_channels * 2, self.MAX_NUM_FILTERS_3D)
        
        in_channels = down_filters[-1][-1]
        self.bottleneck = ConvDropoutNormNonlin(in_channels, in_channels)

        self.ups = []
        in_channels, up_in_channels = down_filters[-1]
        out_channels = down_filters[-1][0]
        self.side_projs = []
        final_channels = -1
        for i in range(len(down_kernel_list)-1, -1, -1):
            stride = stride_list[i]
            if i == 0:
                if direct_supervision:
                    final_channels = class_num
                    has_nonlin = False
                else:
                    final_channels = min(up_in_channels // 2, 2 * class_num)
                    has_nonlin = True
                self.ups.append(UpBlock(up_in_channels, None, final_channels, up_stride=stride, has_nonlin=has_nonlin))
            else:
                self.ups.append(UpBlock(up_in_channels, in_channels, out_channels, up_stride=stride))

            if i <= num_side_loss and i > 0:
                up_scales = np.array([1,1,1])
                for j in range(i):
                    up_scales *= np.array(stride_list[j])
                up_scales = tuple(up_scales.tolist())

                side_proj = nn.Sequential(
                    nn.Conv3d(out_channels, class_num, 1, bias=output_bias),
                    Upsample(scale_factor=up_scales, mode='trilinear'),
                )
                self.side_projs.append(side_proj)
            
            in_channels, up_in_channels = down_filters[i-1]
            out_channels = down_filters[i-1][0]
        
        if not direct_supervision:
            self.class_conv = nn.Conv3d(final_channels, class_num, 1, bias=output_bias)

        # convert layers to nn containers
        self.downs = nn.ModuleList(self.downs)
        self.ups = nn.ModuleList(self.ups)
        self.side_projs = nn.ModuleList(self.side_projs)
            
    def forward(self, x):
        x = self.pre_layer(x)

        skip_feats = []
        ndown = len(self.downs)
        for i in range(ndown - 1):
            x = self.downs[i](x)
            skip_feats.append(x)
        x = self.downs[ndown-1](x)
        
        x = self.bottleneck(x)

        seg_outputs = []
        for i in range(ndown - 1):
            x = self.ups[i](x, skip_feats[ndown-i-2])
            if i >= ndown - 1 - self.num_side_loss:
                proj_idx = i - (ndown - 1 - self.num_side_loss)
                seg_outputs.append(self.side_projs[proj_idx](x))
        
        x = self.ups[ndown-1](x)
        if not self.direct_supervision:
            seg_outputs.append(self.class_conv(x))
        else:
            seg_outputs.append(x)
        
        return seg_outputs[::-1]

if __name__ == '__main__':
    import json
    from torchinfo import summary

    """in_channels = 1
    base_num_filters = 16
    class_num = 2
    down_kernel_list = [[1,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]]
    stride_list = [[1,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2]]
    num_side_loss = 1
    output_bias = False
    direct_supervision = False
    """
    

    conf_file = 'configs/default_config.json'
    with open(conf_file) as fp:
        configs = json.load(fp)
    print('Initialize model...')
    #model = UNet(in_channels, base_num_filters, class_num, down_kernel_list, stride_list, num_side_loss, output_bias=output_bias, direct_supervision=direct_supervision)

    input = torch.randn(2, configs['in_channels'], 128,160,160)
    model = UNet(**configs)
    print(model)

    outputs = model(input)
    for output in outputs:
        print('output size: ', output.size())

    summary(model, input_size=(2,1,128,160,160))
