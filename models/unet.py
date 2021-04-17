#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : unet.py
#   Author       : Yufeng Liu
#   Date         : 2021-03-27
#   Description  : 
#
#================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronet.models.base_model import BaseModel
from neuronet.models.modules.modules import Upsample, ConvDropoutNormNonlin, InitWeights_He, StackedConvLayers

class UNet(BaseModel):
    MAX_NUM_FILTERS_3D = 320
    MAX_FILTERS_2D = 640

    def __init__(self, input_channels=1, base_num_features=8, num_classes=2, num_conv_per_stage=2,
        feat_map_mul_on_downscale=2, conv_op=nn.Conv3d,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs={'eps':1e-5, 'affine':True},
                 dropout_op=nn.Dropout3d, dropout_op_kwargs={'p':0, 'inplace': True},
                 nonlin=nn.LeakyReLU, nonlin_kwargs={"negative_slope":1e-2, 'inplace':True}, deep_supervision=False, dropout_in_localization=False,
                 final_nonlin=nn.Identity, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None, upscale_logits=False, 
                 max_num_features=256, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False, direct_supervision=True):
        
        super(UNet, self).__init__()
        conv_op = eval(conv_op)
        norm_op = eval(norm_op)
        dropout_op = eval(dropout_op)
        nonlin = eval(nonlin)
        final_nonlin = eval(final_nonlin)
        weightInitializer = eval(weightInitializer)
        basic_block = eval(basic_block)

        num_pool = len(pool_op_kernel_sizes)
        # whether manually upscale the lateral supervision
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin()
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.direct_supervision = direct_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            first_stride = pool_op_kernel_sizes[d]

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(
                StackedConvLayers(input_features, output_features, num_conv_per_stage,
                  self.conv_op, self.conv_kwargs, self.norm_op,
                  self.norm_op_kwargs, self.dropout_op,
                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                  first_stride, basic_block=basic_block))

            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)


        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            if u == 0:
                # the first upsampling layer is not skip connected
                input_channels_u = self.conv_blocks_context[-(1+u)].output_channels
                output_channels_u = self.conv_blocks_context[-(1+u)].input_channels
                input_channels_c = output_channels_u * 2
                output_channels_c = self.conv_blocks_context[-(1+u)].input_channels
            elif u == num_pool - 1:
                # the last layer
                input_channels_u = output_channels_c
                if self.direct_supervision:
                    output_channels_u = self.num_classes
                else:
                    output_channels_u = max(base_num_features // 2, 2)
                input_channels_c = output_channels_u
                output_channels_c = output_channels_u
            else:
                input_channels_u = output_channels_c
                output_channels_u = self.conv_blocks_context[-(1+u)].input_channels
                input_channels_c = output_channels_u * 2
                output_channels_c = self.conv_blocks_context[-(1+u)].input_channels

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.tu.append(transpconv(input_channels_u, output_channels_u, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))
          
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(input_channels_c, output_channels_c, num_conv_per_stage,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block)))
            
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))
        
        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(nn.Identity())

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

    def forward(self, x):
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        x = self.conv_blocks_context[-1](x)

        lateral_supervision = self._deep_supervision and self.do_ds
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            if u != len(self.tu) - 1:
                x = torch.cat((x, skips[-(u + 1)]), dim=1)

            if self.direct_supervision and u == len(self.tu) - 1:
                seg_outputs.append(self.final_nonlin(x))
            else:
                x = self.conv_blocks_localization[u](x)
                if lateral_supervision or u == len(self.tu) - 1:
                    seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
        if lateral_supervision:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]


if __name__ == '__main__':
    import json
    from torchinfo import summary

    conf_file = 'configs/default_config.json'
    with open(conf_file) as fp:
        configs = json.load(fp)

    network = UNet(
        **configs
    )
    print(network)

    summary(network, input_size=(2,1,256,512,512))  

