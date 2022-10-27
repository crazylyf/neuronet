#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : fuse_image_with_seg.py
#   Author       : Yufeng Liu
#   Date         : 2021-05-08
#   Description  : 
#
#================================================================
import os, sys, glob
import torch
from collections import Counter
import SimpleITK as sitk
import numpy as np
from skimage import morphology
from skimage.transform import resize
from scipy.ndimage import convolve, median_filter
from scipy.ndimage.interpolation import map_coordinates

from soft_skeletonization import soft_skel

def load_image(imgfile):
    return sitk.GetArrayFromImage(sitk.ReadImage(imgfile))

def save_image(img, imgfile):
    return sitk.WriteImage(sitk.GetImageFromArray(img), imgfile)

def resample(img, new_shape, order=3):
    assert img.ndim == 3, "images should be in 3-dim (z,y,x)"
    shape = np.array(img.shape)
    #assert shape[0] == new_shape[0]
    
    if np.all(shape == new_shape):
        print("No resample required")
        return img
    else:
        print('Resampling...')

    # do sliced 2D image resizing
    new_img = []
    for slice_id in range(shape[0]):
        new_img.append(resize(img[slice_id], new_shape[1:], order=order, mode='edge'))
    new_img = np.stack(new_img, 0)

    if shape[0] != new_shape[0]:
        # resizing in z dimension, code from0 sklearn's resize
        rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
        orig_rows, orig_cols, orig_dim = new_img.shape

        row_scale = float(orig_rows) / rows
        col_scale = float(orig_cols) / cols
        dim_scale = float(orig_dim) / dim 

        map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
        map_rows = row_scale * (map_rows + 0.5) - 0.5 
        map_cols = col_scale * (map_cols + 0.5) - 0.5 
        map_dims = dim_scale * (map_dims + 0.5) - 0.5 

        coord_map = np.array([map_rows, map_cols, map_dims])
        new_img = map_coordinates(new_img, coord_map, order=0, cval=0, mode='nearest')
    
    return new_img

def do_gamma(img, gamma, trunc_thresh=0, normalize=True, epsilon=1e-7):
    minm = img.min()
    rnge = img.max() - minm

    img = (img - minm) / float(rnge + epsilon)
    if trunc_thresh <= 0 or trunc_thresh >= 1:
        img = np.power(img, gamma) * rnge + minm
    else:
        mask = img < trunc_thresh
        img[mask] = np.power(img[mask], gamma)
        img[~mask] = img[~mask] - trunc_thresh + np.power(trunc_thresh, gamma)
        img = img * rnge + minm

    if normalize:
        img = (img - img.min()) / (img.max() - img.min() + epsilon) * 255.
        
    return img
    

def get_image_brain_mapping(map_file='../../data/additional_crops/image_swc_mapper.txt'):
    mapper = {}
    img_dict = {}
    with open(map_file, 'r') as fp:
        for line in fp.readlines():
            line = line.strip()
            if not line: continue
            prefix, swcfile = line.split()
            brain_id = int(swcfile.split('_')[0])
            if brain_id not in img_dict:
                img_dict[brain_id] = [prefix]
            else:
                img_dict[brain_id].append(prefix)
            mapper[prefix] = brain_id

    return mapper, img_dict

def get_single_somas(img_list_file):
    with open(img_list_file) as fp:
        imglist = []
        for line in fp.readlines():
            line = line.strip()
            if not line: continue
            imglist.append(line)
    return set(imglist)

def get_mode_number(img):
    img = img.flatten()
    # to uint8
    img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    v_counter = Counter(img)
    mode_number = sorted(v_counter.items(), key=lambda x:x[-1], reverse=True)[0][0]
    return mode_number

class SegTracer(object):

    def __init__(self, fg_thresh=0.5, alpha=0.5, dil_kernel=(5,13,13), tgamma=0.4, trunc_thresh=0.4, fuse_type='vanilla', bg_mask=False, do_median_filter=False):
        self.fg_thresh = fg_thresh
        self.alpha = alpha
        self.selem = np.ones(dil_kernel, dtype=np.uint8)
        self.tgamma = tgamma
        self.trunc_thresh = trunc_thresh
        self.fuse_type = fuse_type
        self.bg_mask = bg_mask
        self.do_median_filter = do_median_filter

    def fuse_images(self, img, seg):
        if self.bg_mask:
            if self.do_median_filter:
                seg = median_filter(seg, size=(1,3,3), mode='reflect')
            seg_bin = seg > self.fg_thresh
            # keep soma region
            cz,cy,cx = seg.shape[0]//2, seg.shape[1]//2, seg.shape[2]//2
            rz = 15; ry = 40; rx = 40
            seg_bin[cz-rz:cz+rz+1, cy-ry:cy+ry+1, cx-rx:cx+rx+1] = 1
        
            # masking out possible noise
            seg_bin_dil = morphology.dilation(seg_bin, self.selem)
            img_masked = img.copy()
            # get the number of largest number
            img_masked[~seg_bin_dil] = get_mode_number(img)
        else:
            img_masked = img.copy()
        
        
        if self.fuse_type == 'vanilla':
            pass
        elif self.fuse_type == 'aggregate':
            # do average filter to enhance the center voxels
            kernel = np.ones((1,3,3))
            kernel /= kernel.size
            seg = convolve(seg, kernel, mode='reflect', cval=0)
            seg = (seg - seg.min()) / (seg.max() - seg.min() + 1e-7) * 255
        elif self.fuse_type == 'self-aggregate':
            # do average filter to enhance the center voxels
            kernel = np.ones((1,3,3))
            kernel /= kernel.size
            seg_conv = convolve(seg, kernel, mode='reflect', cval=0)
            seg = seg_conv * seg
            seg = (seg - seg.min()) / (seg.max() - seg.min() + 1e-7) * 255
        elif self.fuse_type == 'self-aggregate2':
            # do average filter to enhance the center voxels
            kernel = np.ones((1,3,3))
            kernel /= kernel.size
            seg_conv = convolve(seg, kernel, mode='reflect', cval=0)
            seg = np.power((seg_conv / seg_conv.max()), 5.) * seg
            seg = (seg - seg.min()) / (seg.max() - seg.min() + 1e-7) * 255
        elif self.fuse_type == 'soft-skeleton':
            seg = torch.from_numpy(seg.astype(np.float) / 255.).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                seg = soft_skel(seg, 10, gpu=True).numpy()[0,0]
            seg = (seg - seg.min()) / (seg.max() - seg.min() + 1e-7) * 255
        else:
            raise NotImplementedError

        # fuse with seg
        fused = img_masked * self.alpha + (1 - self.alpha) * seg
        fused = fused.astype(np.uint8)
        return fused

    def resample_and_fuse(self, imgfile, predfile, target_shape, order=3):
        img = load_image(imgfile).astype(np.float32)
        pred = load_image(predfile).astype(np.float32)

        if self.tgamma > 0:
            img = do_gamma(img, gamma=self.tgamma, trunc_thresh=self.trunc_thresh, normalize=True)

        # clip to 0-255
        img = (img - img.min()) / (img.max() - img.min() + 1e-7) * 255

        # get target shape
        img = resample(img, target_shape, order=order)
        pred_res = resample(pred, target_shape, order=order)
        fused = self.fuse_images(img, pred_res)

        return fused

if __name__ == '__main__':
    img_dir = '../../data/additional_crops/crops_tiff'
    pred_dir = '../exp048_brains2'
    target_shape = (256,512,512)
    tgamma = 0
    trunc_thresh = 0.4
    alpha = 0.8
    fuse_type = 'vanilla'
    bg_mask = False
    do_median_filter = False
    phase = 'test'

    if phase == 'par':
        img_list_file = '../par_set_singleSoma.list' 
    elif phase == 'test':
        #img_list_file = '../img_singleSoma.list'
        img_list_file = '../../data/additional_crops/single_soma.list'
    else:
        raise ValueError

    # get the mapping
    mapper, img_dict = get_image_brain_mapping()
    single_soma_set = get_single_somas(img_list_file)
    seg_tracer = SegTracer(tgamma=tgamma, trunc_thresh=trunc_thresh, alpha=alpha, dil_kernel=(1,7,7), fuse_type=fuse_type, bg_mask=bg_mask, do_median_filter=do_median_filter)

    out_dir = os.path.join(pred_dir, f'fused_tg{tgamma:.1f}_alpha{alpha:.1f}_{fuse_type}_bgMask{int(bg_mask)}')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for predfile in sorted(glob.glob(os.path.join(pred_dir, f'debug_*{phase}_pred.tiff'))):
        if phase == 'test':
            prefix = os.path.split(predfile)[-1][6:-15]
        elif phase == 'par':
            prefix = os.path.split(predfile)[-1][6:-14]
        else:
            raise ValueError
        print(prefix)
            
        #if prefix != '2577_6443_2262': continue
        if prefix not in single_soma_set: continue
        print(f'==> Processing image: {prefix}...')

        outfile = os.path.join(out_dir, f'{prefix}.tiff')
        if os.path.exists(outfile): continue
        
        brain_id = mapper[prefix]
        imgfile = os.path.join(img_dir, str(brain_id), f'{prefix}.tiff')
        #imgfile = os.path.join(pred_dir, f'debug_{prefix}_test_img.tiff')
        fused = seg_tracer.resample_and_fuse(imgfile, predfile, target_shape, order=3)

        
        save_image(fused, outfile)
    

