#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : train.py
#   Author       : Yufeng Liu
#   Date         : 2021-03-19
#   Description  : 
#
#================================================================

import os
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn

from neuronet.models import unet
from neuronet.utils import util


parser = argparse.ArgumentParser(
    description='Segmentator for Neuronal Image With Pytorch')
# data specific
parser.add_argument('--train_set', 
                    type=str, help='training set path')
parser.add_argument('--val_set', 
                    type=str, help='val set path')
# training specific
parser.add_argument('--batch_size', default=2, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', action="store_true", 
                    help='Whether use gpu to train model, default True')
parser.add_argument('--amp', action="store_true", 
                    help='Whether to use AMP training, default True')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
# network specific
parser.add_argument('--input_channels', default=1, type=int,
                    help='number of input image channels')
parser.add_argument('--base_num_features', default=8, type=int,
                    help='number of channels for the first layer')

parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


def train():
    # initialize device
    if args.cuda:
        device = util.init_device('cuda')
    else:
        device = util.init_device('cpu')

    # for output folder
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    # dataset preparing
    train_set = ''
    val_set = ''
    train_loader = ''
    val_loader = ''

    # network initialization
    net = unet.UNet()    # explicit initialize

    net = net.to(device)

    # optimizer & loss
    optimizer = optim
    criterion = ''
    criterion = criterion.to(device)

    # training process
    net.train()
    
    for 


if __name__ == '__main__':
    train()






