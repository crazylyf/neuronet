#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : soft_skeletonization.py
#   Author       : Yufeng Liu
#   Date         : 2021-05-27
#   Description  : 
#
#================================================================

import numpy as np
import torch
import torch.nn.functional as F
from skimage.draw import line_nd
import skimage.morphology as morphology

"""
# This is the original version of soft-skeleton, it is not guarantized to be 1-sized skeleton, especially for 3D volumetric image
def soft_erode(img):
    if img.ndim == 5:
        p1 = -F.max_pool3d(-img, (3,1,1), (1,1,1), (1,0,0))
        p2 = -F.max_pool3d(-img, (1,3,1), (1,1,1), (0,1,0))
        p3 = -F.max_pool3d(-img, (1,1,3), (1,1,1), (0,0,1))
        return torch.min(torch.min(p1, p2), p3)
    else:
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1, p2)

def soft_dilate(img):
    if img.ndim == 5:
        return F.max_pool3d(img, (3,3,3), (1,1,1), (1,1,1))
    else:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iter_):
    assert img.max() <= 1 and img.min() >= 0

    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
        print(j, img.mean().item(), img1.mean().item(), skel.mean().item())
    skel = skel
    return skel
"""


def soft_erode_dim(img, dim=0):
    if img.ndim == 5:
        fpool = F.max_pool3d
        kernel, stride, padding = [3,3,3], [1,1,1], [1,1,1]
    else:
        fpool = F.max_pool2d
        kernel, stride, padding = [3,3], [1,1], [1,1]

    kernel[dim] = 1
    padding[dim] = 0
    return -fpool(-img, kernel, stride, padding)

def soft_dilate_dim(img, dim=0):
    if img.ndim == 5:
        fpool = F.max_pool3d
        kernel, stride, padding = [3,3,3], [1,1,1], [1,1,1]
    else:
        fpool = F.max_pool2d
        kernel, stride, padding = [3,3], [1,1], [1,1]
    kernel[dim] = 1
    padding[dim] = 0
    return fpool(img, kernel, stride, padding)

def soft_erode(img):
    if img.ndim == 5:
        return -F.max_pool3d(-img, (3,3,3), (1,1,1), (1,1,1))
    else:
        return -F.max_pool2d(-img, (3,3), (1,1), (1,1))

def soft_dilate(img):
    if img.ndim == 5:
        return F.max_pool3d(img, (3,3,3), (1,1,1), (1,1,1))
    else:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_open_dim(img, dim):
    return soft_dilate_dim(soft_erode_dim(img, dim), dim)

def soft_skel_dim(img, dim, iter_, skel):
    img1 = soft_open_dim(img, dim)
    skel = skel + F.relu(img - img1)
    skel_last = skel.clone()
    for j in range(iter_):
        img = soft_erode_dim(img, dim)
        img1 = soft_open_dim(img, dim)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
        if (skel - skel_last).sum() == 0:
            break
        skel_last = skel
        #print(skel.mean().item())
    print(f'#{j} iters for dim {dim}')

    return img, skel

def soft_skel(img, iter_):
    assert img.max() <= 1 and img.min() >= 0

    ndim = 3 if img.ndim==5 else 2
    skel = 0
    out
    for idim in range(ndim):
        img, skel = soft_skel_dim(img, idim, iter_, skel)
        

    return skel

def synthesize_image(shape=(512,512), maxv=255, start_pt=(120,120), end_pt=(120,480)):
    if (start_pt is None) or (end_pt is None):
        raise NotImplementedError

    img = np.zeros(shape, dtype=np.float)
    lin = line_nd(start_pt, end_pt, )
    img[lin] = maxv
    
    selem = np.ones((5,5), dtype=np.uint8)
    img = morphology.dilation(img, selem)
    return img


if __name__ == '__main__':
    from file_io import load_image, save_image
    import os, glob

    imgpath = '/home/lyf/Temp/exp040/tg0.4_retainStats/fused_tg0.0_alpha0.8_vanilla_bgMask0'
    iter_ = 20
    use_gpu = False

    outpath = os.path.join(imgpath, 'skel')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    for imgfile in glob.glob(os.path.join(imgpath, "[1-9]*[0-9].tiff")):
        imgname = os.path.split(imgfile)[-1]
        print(f'==> Processing for image: {imgname}')
        
        img = load_image(imgfile)
        #img = synthesize_image()
        # convert to [0,1] and float 
        img = torch.from_numpy(img.astype(np.float32) / 255.).unsqueeze(0).unsqueeze(0)   # assume 255 maximal
        if use_gpu:
            device = torch.device('cuda:0')
            img = img.to(device)
        vmax, vmin = img.max().item(), img.min().item()
        vmean, vstd = img.mean().item(), img.std().item()
        # do thresholding
        bkg_thresh = vmean + 0.5 * max(10./255, vstd)
        print(img.shape, vmax, vmin, vmean, vstd, bkg_thresh)
        img[img >= bkg_thresh] = 1
        img[img < bkg_thresh] = 0
        with torch.no_grad():
            skel = soft_skel(img, iter_).cpu().numpy()[0,0]
        print(skel.shape, skel.max(), skel.min(), skel.mean(), '\n')
        skel = (skel - skel.min()) / (skel.max() - skel.min() + 1e-7)
        skel = (skel * 255).astype(np.uint8)
        outfile = os.path.join(outpath, imgname)
        save_image(outfile, skel)
        break

