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
from neuronet.datasets.swc_processing import trim_swc, swc_to_image, swc_to_connection

# To avoid the recursionlimit error, maybe encountered in trim_swc
sys.setrecursionlimit(30000)

class GenericDataset(tudata.Dataset):

    def __init__(self, split_file, phase='train', imgshape=(256,512,512)):
        self.data_list = self.load_data_list(split_file, phase)
        self.imgshape = imgshape

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
        tree = trim_swc(tree, self.imgshape, True)
        mask, lab = swc_to_connection(tree, 3, 1, self.imgshape, True)
        
        return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(lab.astype(np.uint8)), torch.from_numpy(mask.astype(np.bool)), imgfile, swcfile


if __name__ == '__main__':
    split_file = '../data/task0001_17302/data_splits.pkl'
    idx = 2
    imgshape = (256,510,510)
    dataset = GenericDataset(split_file, 'train', imgshape)
    img, lab = dataset.pull_item(idx)

