#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : dataset.py
#   Author       : Yufeng Liu
#   Date         : 2021-04-01
#   Description  : 
#
#================================================================

import numpy as np
import pickle
import torch.utils.data as tudata
import SimpleITK as sitk
import torch
import sys

from swc_handler import parse_swc
from neuronet.augmentation.generic_augmentation import InstanceAugmentation
from neuronet.datasets.swc_processing import trim_swc, swc_to_image, swc_to_connection, soma_labelling, trim_out_of_box

# To avoid the recursionlimit error, maybe encountered in trim_swc
sys.setrecursionlimit(30000)

class GenericDataset(tudata.Dataset):

    def __init__(self, split_file, phase='train', imgshape=(256,512,512), label_soma=True):
        self.data_list = self.load_data_list(split_file, phase)
        self.imgshape = imgshape
        self.label_soma = label_soma

        # augmentations
        self.augment = InstanceAugmentation(p=0.2, imgshape=imgshape, phase=phase)
    
    @staticmethod
    def load_data_list(split_file, phase):
        with open(split_file, 'rb') as fp:
            data_dict = pickle.load(fp)
        return data_dict[phase]

    def __getitem__(self, index):
        img, gt, mask, imgfile, swcfile = self.pull_item(index)
        return img, gt, mask, imgfile, swcfile

    def __len__(self):
        return len(self.data_list)

    def pull_item(self, index):
        imgfile, swcfile, spacing = self.data_list[index]
        # parse, image should in [c,z,y,x] format
        img = np.load(imgfile)['data']
        if img.ndim == 3:
            img = img[None]
        tree = parse_swc(swcfile)
        # random augmentation
        img, tree, _ = self.augment(img, tree, spacing)
        # convert swc to image
        # firstly trim_swc via deleting out-of-box points
        tree = trim_out_of_box(tree, self.imgshape, True)
        mask, lab = swc_to_connection(tree, 3, 1, self.imgshape, flipy=True)
        if label_soma:
            mask = soma_labelling(mask, z_ratio=0.3, r=3, thresh=220, label=1)

        #print('Image: ', img.mean(), img.min(), img.max(), img.shape)
        #print('Mask: ', mask.mean(), mask.min(), mask.max(), mask.shape)
        #print('Lab: ', lab.mean(), lab.min(), lab.max(), lab.shape)
        
        return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(lab.astype(np.uint8)), torch.from_numpy(mask.astype(np.bool)), imgfile, swcfile


if __name__ == '__main__':
    split_file = '../data/task0001_17302/data_splits.pkl'
    idx = 2
    imgshape = (256,510,510)
    dataset = GenericDataset(split_file, 'train', imgshape)
    img, lab = dataset.pull_item(idx)

