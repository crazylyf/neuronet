#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : zmq_inference.py
#   Author       : Yufeng Liu
#   Date         : 2021-11-29
#   Description  : 
#
#================================================================

import os, sys
import argparse
import numpy as np
import time
import json
import math
import zmq

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from neuronet.models import unet
from neuronet.utils import util
from neuronet.utils.image_util import unnormalize_normal, normalize_normal, do_gamma
from neuronet.evaluation.multi_crop_evaluation import MostFitCropEvaluation
from file_io import load_image, save_image

parser = argparse.ArgumentParser(
    description='Segmentator for Neuronal Image With Pytorch')
# training specific
parser.add_argument('--image_shape', default='128,128,128', type=str,
                    help='Image shape at each dimension')
parser.add_argument('--amp', action="store_true",
                    help='Whether to use AMP training, default True')
parser.add_argument('--enhancement', default='DTGT', type=str,
                    help='enhancement type, e.g. DTGT, GT, NONE, CLAHE, etc.')
parser.add_argument('--deterministic', action='store_true',
                    help='run in deterministic mode')
parser.add_argument('--seed', default=1025, type=int,
                    help='Random seed value')
parser.add_argument('--eval_flip', action='store_true',
                    help='Whether to do flipped ensembling')
parser.add_argument('--checkpoint', default='', type=str,
                    help='Saved checkpoint')
# network specific
parser.add_argument('--net_config', default="./models/configs/default_config.json",
                    type=str,
                    help='json file defining the network configurations')

args = parser.parse_args()


def pad_levels2(image, imgshape, minshape, ds_ratios, padding='OppositePad'):
    """
    All shapes and ratios are in order [Z,Y,X].
    Personally, I would like to use an novel padding style, which I refer to `OppositePad`. 
    This padding means that we pad the current side with boundary values of opposite side. In
    this way, we want to realize:
        - padding with block with similar statistics
        - does not introduce artifact by replicate or duplicate like paddings.
    """
    # small block
    cur_is = np.array(image.shape)
    psize = np.maximum(np.array(minshape) - cur_is, 0)
    cur_is = cur_is + psize
    # median block 
    p2 = []
    for si, sj, dr in zip(cur_is, imgshape, ds_ratios):
        if si > sj:
            p2.append(0)
        else:
            # the image is not that big, we can send to prediction directly. Before prediction
            # make sure image shape is dividable by ds ratio
            ss = int(math.ceil(si / dr) * dr) - si
            p2.append(ss)
    p2 = np.array(p2)
    # overall
    pp = psize + p2
    if padding == 'OppositePad':
        pad_image = image.copy() 
        # to be implemented


    return pad_image
    
def pad_levels1(image, imgshape, padding='OppositePad'):
    """
    All shapes and ratios are in order [Z,Y,X].
    Personally, I would like to use an novel padding style, which I refer to `OppositePad`. 
    This padding means that we pad the current side with boundary values of opposite side. In
    this way, we want to realize:
        - padding with block with similar statistics
        - does not introduce artifact by replicate or duplicate like paddings.
    """
    # small block
    orig_is = np.array(image.shape)
    psize = np.maximum(np.array([1,*imgshape]) - orig_is, 0)
    cur_is = orig_is + psize
    pad_image = np.zeros(cur_is, dtype=image.dtype)
    if padding == 'OppositePad':
        for zi in range(int(math.ceil(cur_is[1] / orig_is[1]))):
            zs = zi * orig_is[1]
            ze = min(orig_is[1], cur_is[1] - zs)
            for yi in range(int(math.ceil(cur_is[2] / orig_is[2]))):
                ys = yi * orig_is[2]
                ye = min(orig_is[2], cur_is[2] - ys)
                for xi in range(int(math.ceil(cur_is[3] / orig_is[3]))):
                    xs = xi * orig_is[3]
                    xe = min(orig_is[3], cur_is[3] - xs)
                    pad_image[:,zs:zs+ze,ys:ys+ye,xs:xs+xe] = image[:,:ze,:ye,:xe].copy()
                    
    else:
        raise NotImplementedError(f"Padding type {padding} is not supported yet!")

    return pad_image, psize


def forward_inference(image, model):
    if args.amp:
        with autocast():
            with torch.no_grad():
                pred = model(image)
    else:
        with torch.no_grad():
            pred = model(image)
    return pred

def inference(model, image):
    # the input image is in range [0,255], with type uint8, and shape [1,Z,Y,X]
    # We should firstly convert to float
    image = image.astype(np.float32)
    # standardize
    image = normalize_normal(image, mask=None)
    # do gamma transformation
    if args.enhancement == 'NONE':
        print(f'No enhancement required!')
    elif args.enhancement == 'DTGT':
        print(f'Apply DTGT...')
        image = do_gamma(image, gamma=0.4, trunc_thresh=0.216, retain_stats=True)
    else:
        raise NotImplementedError(f"Enhancement type {args.enhancement} is not implemented!")
    # padding image
    pad_image, psize = pad_levels1(image, args.image_shape, padding='OppositePad')
    pis = pad_image.shape
    # image to device
    img_d = torch.from_numpy(pad_image).to(args.device)[None] # [B,C,Z,Y,X]
    
    # initialize multi-crop-evaluation
    mce = MostFitCropEvaluation(args.image_shape, args.ds_ratios)
    n_ens = 4 if args.eval_flip else 1
    for ie in range(n_ens):
        if ie == 0:
            crops, crop_sizes, lab_crops = mce.get_image_crops(img_d[0], None)
        elif ie == 1:
            crops, crop_sizes, lab_crops = mce.get_image_crops(torch.flip(img_d[0], [2]), None)
        elif ie == 2:
            crops, crop_sizes, lab_crops = mce.get_image_crops(torch.flip(img_d[0], [3]), None)
        elif ie == 3:
            crops, crop_sizes, lab_crops = mce.get_image_crops(torch.flip(img_d[0], [2,3]), None)
        # forward
        preds = []
        for i in range(len(crops)):
            pred = forward_inference(crops[i][None], model)
            preds.append(pred[0][0])   # last prediction, in shape[B,1,Z,Y,X]
        
        # merge the crops
        pred = mce.get_pred_from_crops(img_d[0,0].shape, preds, crop_sizes)
        if ie == 0:
            avg_pred = pred
        elif ie == 1:
            avg_pred += torch.flip(pred, [2])   # bug in train.py
        elif ie == 2:
            avg_pred += torch.flip(pred, [3])
        elif ie == 3:
            avg_pred += torch.flip(pred, [2,3])
        else:
            raise ValueError

        del crops, preds
 
    # averaging through all predictions
    avg_pred /= n_ens   # avg_pred in shape[C,Z,Y,X]
    # convert to probability space
    probs = F.softmax(avg_pred, dim=0)[1]
    # discard padded region
    probs = probs[:pis[1]-psize[1],:pis[2]-psize[2],:pis[3]-psize[3]].contiguous().to(torch.device('cpu')).numpy().astype(np.float32)   # psize in shape[1,Z,Y,X]

    return probs
    
def fuse_image(img, seg, alpha=0.8, eps=1e-7):
    img = img.astype(np.float32)
    seg = seg.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + eps)
    seg = (seg - seg.min()) / (seg.max() - seg.min() + eps)
    fused = alpha * img + (1 - alpha) * seg
    # rescale to [0, 255]
    fused = (fused * 255).astype(np.uint8)
    return fused

def main(): 
    gpu_id = 0
    args.device = util.init_device(gpu_id)
    if args.deterministic:
        util.set_deterministic(deterministic=True, seed=args.seed)

    # Network
    with open(args.net_config) as fp:
        net_configs = json.load(fp)
        print('Network configs: ', net_configs)
        model = unet.UNet(**net_configs)
        print('\n' + '='*10 + 'Network Structure' + '='*10)
        print(model)
        print('='*30 + '\n')

    
    # get the network downsizing informations
    ds_ratios = np.array([1,1,1])
    for stride in net_configs['stride_list']:
        ds_ratios *= np.array(stride)
    args.ds_ratios = tuple(ds_ratios.tolist())

    model = model.to(args.device)
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location={'cuda:0':f'cuda:{gpu_id}'})
    model.load_state_dict(checkpoint)
    del checkpoint

    args.image_shape = tuple(map(int, args.image_shape.split(',')))
    # Print out the args information
    print('Arguments are: ')
    print(f'    {args}')

    
    # intialize socket
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("ipc:///tmp/feeds")

    # receive image from client
    while True:
        recved = socket.recv_json()
        dtype = recved['type']
        dimX = recved['dimX']
        dimY = recved['dimY']
        dimZ = recved['dimZ']
        imgraw = recved['img']
        image = np.array(imgraw, dtype=np.uint8).reshape(1, dimZ, dimY, dimX)
        #save_image('orig.tiff', image[0])

        # Execute inference
        seg = inference(model, image)
        print(f'Statistics of segmentation: {seg.mean()}, {seg.std()}, {seg.min()}, {seg.max()}')
        # convert to uint8 and send back
        seg = (seg * 255).astype(np.uint8)
        # fusing with original image
        fused = fuse_image(image[0], seg, 0.8)
        #save_image('seg.tiff', seg)
        fused = fused.reshape(-1).tolist()
        repl = {}
        repl['img'] = fused
        socket.send_json(repl)


if __name__ == '__main__':
    main()

    
    
    
    
    
