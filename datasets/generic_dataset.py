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

from swc_handler import parse_swc

class GenericDataset(tudata.Dataset):

    def __init__(self, split_file, phase='train'):
        self.data_list = self.load_data_list(split_file, phase)

        # augmentations
        if phase == 'train':
            self.augment = None
        elif phase == 'val':
            self.augment = None
        elif phase == 'test':
            self.augment = None
        else:
            raise NotImplementedError('phase should be train/val/test')
    
    @staticmethod
    def load_data_list(split_file, phase):
        with open(split_file, 'rb') as fp:
            data_dict = pickle.load(fp)
        return data_dict[phase]

    def __getitem__(self, index):
        img, gt = self.pull_item(index)
        return img, gt

    def __len__(self):
        return len(self.data_list)

    def pull_item(self, index):
        imgfile, swcfile = self.data_list[index]
        # parse, image should in [c,z,y,x] format
        img = np.load(imgfile)['data']
        if img.ndim == 3:
            img = img[None]
        swc = parse_swc(swcfile)
        # random augmentation
        img, swc = self.augment(img, swc)
        # convert swc to image
        lab = swc_to_image(swc)
        
        return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(lab.astype(np.int))

