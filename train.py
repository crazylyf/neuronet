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
import torch.utils.data as tudata

from neuronet.models import unet
from neuronet.utils import util
from neuronet.datasets.generic_dataset import GenericDataset


parser = argparse.ArgumentParser(
    description='Segmentator for Neuronal Image With Pytorch')
# data specific
parser.add_argument('--data_file', 
                    type=str, help='data file')
# training specific
parser.add_argument('--batch_size', default=2, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--image_shape', default='256,512,512', type=str,
                    help='Input image shape')
parser.add_argument('--cuda', action="store_true", 
                    help='Whether use gpu to train model, default True')
parser.add_argument('--amp', action="store_true", 
                    help='Whether to use AMP training, default True')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.99, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=3e-5, type=float,
                    help='Weight decay for SGD')
# network specific
parser.add_argument('--net_config', default="./models/configs/default_config.json",
                    type=str,
                    help='json file defining the network configurations')

parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    """
    poly_lr policy as the same as nnUNet
    """
    return initial_lr * (1 - epch / max_epochs)**exponent

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
    imgshape = tuple(map(int, args.image_shape.split(',')))
    train_set = GenericDataset(args.data_file, phase='train', imgshape)
    val_set = GenericDataset(args.data_file, phase='val', imgshape)
    train_loader = tudata.DataLoader(train_set, args.batch_size, 
                                    num_workers=args.num_workers, 
                                    shuffle=True, pin_memory=True)
    val_loader = tudata.DataLoader(val_set, args.batch_size, 
                                    num_workers=args.num_workers, 
                                    shuffle=False, pin_memory=True)

    # network initialization
    with open(args.net_config) as fp:
        net_configs = json.load(fp)
        model = unet.UNet(**net_configs)
        print('\n' + '='*10 + 'Network Structure' + '='*10)
        print(model)
        print('='*30 + '\n')

    model = model.to(device)

    # optimizer & loss
    optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # training process
    net.train()
    
    for 


if __name__ == '__main__':
    train()






