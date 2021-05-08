#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : multi_crop_evaluation.py
#   Author       : Yufeng Liu
#   Date         : 2021-04-29
#   Description  : 
#
#================================================================

import math
import torch
import numpy as np
import torch.nn.functional as F

class GenericMultiCropEvaluation(object):
    def __init__(self, patch_size, divid=(2**4,2**5,2**5), pad_value='mean'):
        self.patch_size = patch_size
        self.divid = divid
        self.pad_value = pad_value

    def get_divid_shape(self, dim_size, divid):
        new_size = int(math.ceil(dim_size / divid) * divid)
        return new_size

    def get_pad_value(self, img):
        if self.pad_value == 'min':
            pad_value, _ = img.reshape((img.shape[0], -1)).min(dim=1)
        elif self.pad_value == 'max':
            pad_value, _ = img.reshape((img.shape[0], -1)).max(dim=1)
        elif self.pad_value == 'mean':
            pad_value, _ = img.reshape((img.shape[0], -1)).mean(dim=1)
        elif isinstance(self.pad_value, int) or isinstance(self.pad_value, float):
            pad_value = self.pad_value
        elif isinstance(self.pad_value, tuple) or isinstance(self.pad_value, list):
            pad_value = self.pad_value
        else:
            raise ValueError("pad_value is incorrect!")
        return pad_value

    def run_evalute(self, img):
        pass

class NonOverlapCropEvaluation(GenericMultiCropEvaluation):
    def __init__(self, patch_size, divid=(2**4,2**5,2**5), pad_value='min'):
        super(NonOverlapCropEvaluation, self).__init__(patch_size, divid, pad_value)

    def get_crop_sizes(self, imgshape):
        crop_sizes = []
        pads = []
        for pi, si, di in zip(self.patch_size, imgshape, self.divid):
            crop_size = self.get_divid_shape(pi, di)
            ncrop = int(math.ceil(si / crop_size))
            crop_sizes.append(crop_size)
            pads.append(crop_size * ncrop - si)
            
        return crop_sizes, pads

    def get_image_crops(self, img, lab=None):
        # image in shape [c, z, y, x]
        imgshape = img[0].shape

        pad_value = self.get_pad_value(img)
        crops = []
        lab_crops = None
        if lab is not None:
            lab_crops = []

        crop_sizes, pads = self.get_crop_sizes(imgshape)
        (size_z, size_y, size_x) = crop_sizes
        for zi in range(int(math.ceil(imgshape[0] / size_z))):
            zs, ze = zi * size_z, (zi+1) * size_z
            for yi in range(int(math.ceil(imgshape[1] / size_y))):
                ys, ye = yi * size_y, (yi+1) * size_y
                for xi in range(int(math.ceil(imgshape[2] / size_x))):
                    xs, xe = xi * size_x, (xi+1) * size_x
                    crop = img[:, zs:ze, ys:ye, xs:xe]
                    if crop.shape != tuple(crop_sizes):
                        new_crop = torch.ones((img.shape[0], *crop_sizes), dtype=img.dtype, device=img.device) * pad_value
                        cropz, cropy, cropx = crop[0].shape
                        new_crop[:, :cropz, :cropy, :cropx] = crop
                        crops.append(new_crop)

                        # for lab
                        if lab is not None:
                            new_lab_crop = torch.zeros(crop_sizes, dtype=lab.dtype, device=lab.device)
                            new_lab_crop[:cropz, :cropy, :cropx] = lab[zs:ze, ys:ye, xs:xe]
                            lab_crops.append(new_lab_crop)
                    else:
                        crops.append(crop)
                        if lab is not None:
                            lab_crops.append(lab[zs:ze, ys:ye, xs:xe])

        return crops, crop_sizes, lab_crops
        
    def get_pred_from_crops(self, imgshape, preds, crop_sizes):
        """
        imgshape: (z_dim,y_dim,x_dim)
        preds: [(C,Z,Y,X), ...]
        crop_sizes: (z_size,y_size,x_size)
        """
        # Number of crops in each dimension
        ncrops = [int(math.ceil(imgshape[i]/preds[0][0].shape[i])) for i in range(3)]
        full_size = (preds[0].shape[0], ncrops[0]*crop_sizes[0], ncrops[1]*crop_sizes[1], ncrops[2]*crop_sizes[2])
        pred = torch.zeros(full_size, dtype=preds[0].dtype, device=preds[0].device)
        size_z, size_y, size_x = crop_sizes

        for zi in range(int(math.ceil(imgshape[0] / size_y))):
            zs, ze = zi * size_z, (zi+1) * size_z
            for yi in range(int(math.ceil(imgshape[1] / size_y))):
                ys, ye = yi * size_y, (yi+1) * size_y
                for xi in range(int(math.ceil(imgshape[2] / size_x))):
                    xs, xe = xi * size_x, (xi+1) * size_x
                    idx = xi + yi * ncrops[1] + zi * ncrops[1] * ncrops[2]
                    pred[:, zs:ze, ys:ye, xs:xe] = preds[idx]
        
        return pred[:,:imgshape[0],:imgshape[1],:imgshape[2]]


class MostFitCropEvaluation(GenericMultiCropEvaluation):
    def __init__(self, patch_size, divid=(2**4,2**5,2**5), pad_value='min'):
        super(MostFitCropEvaluation, self).__init__(patch_size, divid, pad_value)

    def get_crop_sizes(self, imgshape):
        crop_sizes = []
        pads = []
        for pi, si, di in zip(self.patch_size, imgshape, self.divid):
            if pi >= si:
                crop_size = self.get_divid_shape(si, di)
                crop_sizes.append(crop_size)
                pads.append(crop_size - imgshape)
            else:
                crop_size = self.get_divid_shape(pi, di)
                ncrop = int(math.ceil(si / crop_size))
                crop_sizes.append(crop_size)
                pads.append(crop_size * ncrop - si)
            
        return crop_sizes, pads

    def get_image_crops(self, img, lab=None):
        # image in shape [c, z, y, x]
        imgshape = img[0].shape
        crop_sizes, pseudo_pads = self.get_crop_sizes(imgshape)
        (size_z, size_y, size_x) = crop_sizes

        # pre-padding images
        padding = []
        for i, crop_size in enumerate(crop_sizes):
            if imgshape[i] < crop_size:
                padding.append(0)
                padding.append(crop_size - imgshape[i])
            else:
                padding.append(0)
                padding.append(0)
        #print(padding)

        # NOTE: for multi-modulity image, this may problematic!! to be fixed later
        pad_value = self.get_pad_value(img)
        if isinstance(pad_value, torch.Tensor):
            pad_value = float(pad_value[0])
        
        pad_img = F.pad(img[None], padding, mode='constant', value=pad_value)[0]  # 3D padding requires 5D input
        nshape = pad_img[0].shape
        lab_crops = None
        if lab is not None:
            lab_crops = []
            pad_lab = F.pad(lab[None][None], padding, mode='constant', value=0)[0,0]
        crops = []
        
        for zi in range(int(math.ceil(nshape[0] / size_z))):
            zs, ze = zi * size_z, (zi+1) * size_z
            if ze > nshape[0]:
                ze = None
                zs = -size_z
            for yi in range(int(math.ceil(nshape[1] / size_y))):
                ys, ye = yi * size_y, (yi+1) * size_y
                if ye > nshape[1]:
                    ye = None
                    ys = -size_y

                for xi in range(int(math.ceil(nshape[2] / size_x))):
                    xs, xe = xi * size_x, (xi+1) * size_x
                    if xe > nshape[2]:
                        xe = None
                        xs = -size_x
                    #print(zi,yi,xi,zs,ze,ys,ye,xs,xe,pad_img.shape)
                    crop = pad_img[:, zs:ze, ys:ye, xs:xe]
                    crops.append(crop)
                    if lab is not None:
                        lab_crop = pad_lab[zs:ze, ys:ye, xs:xe]
                        lab_crops.append(lab_crop)
        
        return crops, crop_sizes, lab_crops
        
    def get_pred_from_crops(self, imgshape, preds, crop_sizes):
        """
        imgshape: (z_dim,y_dim,x_dim)
        preds: [(C,Z,Y,X), ...]
        crop_sizes: (z_size,y_size,x_size)
        """
        # Number of crops in each dimension
        ncrops = [int(math.ceil(imgshape[i]/preds[0][0].shape[i])) for i in range(3)]
        
        size_zf = crop_sizes[0] if crop_sizes[0] >= imgshape[0] else imgshape[0]
        size_yf = crop_sizes[1] if crop_sizes[1] >= imgshape[1] else imgshape[1]
        size_xf = crop_sizes[2] if crop_sizes[2] >= imgshape[2] else imgshape[2]
        full_size = (preds[0].shape[0], size_zf, size_yf, size_xf)
        pred = torch.zeros(full_size, dtype=preds[0].dtype, device=preds[0].device)
        size_z, size_y, size_x = crop_sizes

        for zi in range(ncrops[0]):
            zs, ze = zi * size_z, (zi+1) * size_z
            if zi == ncrops[0] - 1:
                ze = None
                zs = -size_z
            for yi in range(ncrops[1]):
                ys, ye = yi * size_y, (yi+1) * size_y
                if yi == ncrops[1] - 1:
                    ye = None
                    ys = -size_y

                for xi in range(ncrops[2]):
                    xs, xe = xi * size_x, (xi+1) * size_x
                    if xi == ncrops[2] - 1:
                        xe = None
                        xs = -size_x
                    #NOTE: it is definitely better to combines results for overlap regions. to be optimized later.    
                    idx = xi + yi * ncrops[1] + zi * ncrops[1] * ncrops[2]
                    pred[:, zs:ze, ys:ye, xs:xe] = preds[idx]
        
        return pred[:,:imgshape[0],:imgshape[1],:imgshape[2]]


if __name__ == '__main__':
    patch_size = (16,25,25)
    divid = (2**2,2**2,2**2)
    img = torch.rand((1,34,49,36))
    lab = torch.rand((34,49,36)) > 0.5

    imgshape = lab.shape
    noce = MostFitCropEvaluation(patch_size, divid=divid)
    crops, crop_sizes, lab_crops = noce.get_image_crops(img, lab)
    print(f'Crop size: ', crop_sizes)
    print(f'Number of crops: ', len(crops))
    for i, crop, lab_crop in zip(range(len(crops)), crops, lab_crops):
        print(i, crop.size(), lab_crop.size())
    
    pred = noce.get_pred_from_crops(imgshape, crops, crop_sizes)
    print(pred.shape)

