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
import time
import json
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.utils.data as tudata
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.distributed as distrib
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from neuronet.models import unet
from neuronet.utils import util
from neuronet.utils.image_util import unnormalize_normal
from neuronet.datasets.generic_dataset import GenericDataset
from neuronet.loss.dice_loss import BinaryDiceLoss

import path_util


parser = argparse.ArgumentParser(
    description='Segmentator for Neuronal Image With Pytorch')
# data specific
parser.add_argument('--data_file', default='./data/task0001_17302/data_splits.pkl',
                    type=str, help='dataset split file')
# training specific
parser.add_argument('--batch_size', default=2, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--image_shape', default='256,512,512', type=str,
                    help='Input image shape')
parser.add_argument('--cpu', action="store_true", 
                    help='Whether use gpu to train model, default True')
parser.add_argument('--amp', action="store_true", 
                    help='Whether to use AMP training, default True')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.99, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=3e-5, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--max_epochs', default=200, type=int,
                    help='maximal number of epochs')
parser.add_argument('--step_per_epoch', default=200, type=int,
                    help='step per epoch')
parser.add_argument('--deterministic', action='store_true',
                    help='run in deterministic mode')
parser.add_argument('--test_frequency', default=3, type=int,
                    help='frequency of testing')
parser.add_argument('--local_rank', default=-1, type=int, metavar='N', 
                    help='Local process rank')  # DDP required
parser.add_argument('--seed', default=1025, type=int,
                    help='Random seed value')
parser.add_argument('--checkpoint', default='', type=str,
                    help='Saved checkpoint')
# network specific
parser.add_argument('--net_config', default="./models/configs/default_config.json",
                    type=str,
                    help='json file defining the network configurations')

parser.add_argument('--save_folder', default='exps/temp',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()



def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    """
    poly_lr policy as the same as nnUNet
    """
    return initial_lr * (1 - epoch / max_epochs)**exponent

# test only function
def crop_data(img, lab):
    #img = img[:,:,96:160,192:320,192:320]
    #lab = lab[:,96:160,192:320,192:320]
    return img, lab

def ddp_print(content):
    if args.is_master:
        print(content)

def save_image_in_training(imgfiles, img, lab, logits, epoch, phase, idx):
    imgfile = imgfiles[idx]
    prefix = path_util.get_file_prefix(imgfile)
    with torch.no_grad():
        img_v = (unnormalize_normal(img[idx].numpy())[0]).astype(np.uint8)
        lab_v = (unnormalize_normal(lab[[idx]].numpy().astype(np.float))[0]).astype(np.uint8)
        
        logits = F.softmax(logits, dim=1).to(torch.device('cpu'))
        log_v = (unnormalize_normal(logits[idx,[1]].numpy())[0]).astype(np.uint8)
        if phase == 'train':
            out_img_file = f'debug_epoch{epoch}_{prefix}_{phase}_img.tiff'
            out_lab_file = f'debug_epoch{epoch}_{prefix}_{phase}_lab.tiff'
            out_pred_file = f'debug_epoch{epoch}_{prefix}_{phase}_pred.tiff'
        else:
            out_img_file = f'debug_{prefix}_{phase}_img.tiff'
            out_lab_file = f'debug_{prefix}_{phase}_lab.tiff'
            out_pred_file = f'debug_{prefix}_{phase}_pred.tiff'

        sitk.WriteImage(sitk.GetImageFromArray(img_v), os.path.join(args.save_folder, out_img_file))
        sitk.WriteImage(sitk.GetImageFromArray(lab_v), os.path.join(args.save_folder, out_lab_file))
        sitk.WriteImage(sitk.GetImageFromArray(log_v), os.path.join(args.save_folder, out_pred_file))


def get_forward(img_d, lab_d, crit_ce, crit_dice, model):
    logits = model(img_d)
    loss_ce = crit_ce(logits, lab_d.long())
    loss_dice = crit_dice(logits, lab_d)
    loss = loss_ce + loss_dice
    return loss_ce, loss_dice, loss, logits


def validate(model, val_loader, device, crit_ce, crit_dice, epoch, debug=True):
    model.eval()
    avg_ce_loss = 0
    avg_dice_loss = 0
    max_show = 20
    num_show = 0
    for img,lab,imgfiles,swcfiles in val_loader:
        img, lab = crop_data(img, lab)
        img_d = img.to(device)
        lab_d = lab.to(device)
        
        with torch.no_grad():
            loss_ce, loss_dice, loss, logits = get_forward(img_d, lab_d, crit_ce, crit_dice, model)
            del img_d

        avg_ce_loss += loss_ce
        avg_dice_loss += loss_dice

        if debug and args.is_master:
            for debug_idx in range(img.size(0)):
                num_show += 1
                if num_show > max_show:
                    break
                save_image_in_training(imgfiles, img, lab, logits, epoch, 'val', debug_idx)

    avg_ce_loss /= len(val_loader)
    avg_dice_loss /= len(val_loader)
    
    

    model.train()

    return avg_ce_loss, avg_dice_loss

def train():

    if args.deterministic:
        util.set_deterministic(deterministic=True, seed=args.seed)

    # for output folder
    if args.is_master and not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    # dataset preparing
    imgshape = tuple(map(int, args.image_shape.split(',')))
    train_set = GenericDataset(args.data_file, phase='train', imgshape=imgshape)
    val_set = GenericDataset(args.data_file, phase='val', imgshape=imgshape)
    ddp_print(f'Number of train and val samples: {len(train_set)}, {len(val_set)}')
    # distributedSampler
    train_sampler = DistributedSampler(train_set, shuffle=True)
    val_sampler = DistributedSampler(val_set, shuffle=False)

    train_loader = tudata.DataLoader(train_set, args.batch_size, 
                                    num_workers=args.num_workers, 
                                    shuffle=False, pin_memory=True, 
                                    sampler=train_sampler,
                                    drop_last=True, 
                                    worker_init_fn=util.worker_init_fn)
    val_loader = tudata.DataLoader(val_set, args.batch_size, 
                                    num_workers=args.num_workers, 
                                    sampler=val_sampler,
                                    shuffle=False, pin_memory=True,
                                    drop_last=True, 
                                    worker_init_fn=util.worker_init_fn)
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    # network initialization
    with open(args.net_config) as fp:
        net_configs = json.load(fp)
        model = unet.UNet(**net_configs)
        ddp_print('\n' + '='*10 + 'Network Structure' + '='*10)
        ddp_print(model)
        ddp_print('='*30 + '\n')

    #import ipdb; ipdb.set_trace()
    model = model.to(args.device)
    if args.checkpoint:
        # load checkpoint
        print(f'Loading checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location={'cuda:0':f'cuda:{args.local_rank}'})
        model.load_state_dict(checkpoint.module.state_dict())
    
    # convert to distributed data parallel model
    model = DDP(model, device_ids=[args.local_rank],
                output_device=args.local_rank, find_unused_parameters=True)

    # optimizer & loss
    if args.checkpoint:
        args.lr /= 10.
    #optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay, amsgrad=True)
    crit_ce = nn.CrossEntropyLoss().to(args.device)
    crit_dice = BinaryDiceLoss(smooth=1e-5).to(args.device)

    # training process
    model.train()

    
    t0 = time.time()
    grad_scaler = GradScaler()
    debug = True
    debug_idx = 0
    best_loss_dice = 1.0e10
    for epoch in range(args.max_epochs):
        avg_loss_ce = 0
        avg_loss_dice = 0
        for it in range(args.step_per_epoch):
            try:
                img, lab, imgfiles, swcfiles = next(train_iter)
            except StopIteration:
                # let all processes sync up before starting with a new epoch of training
                distrib.barrier()
                # reset the random seed, to avoid np.random & dataloader problem
                np.random.seed(args.seed + epoch)
                
                train_iter = iter(train_loader)
                img, lab, imgfiles, swcfiles = next(train_iter)

            # center croping for debug, 64x128x128 patch
            img, lab = crop_data(img, lab)

            img_d = img.to(args.device)
            lab_d = lab.to(args.device)
            
            optimizer.zero_grad()
            if args.amp:
                with autocast():
                    loss_ce, loss_dice, loss, logits = get_forward(img_d, lab_d, crit_ce, crit_dice, model)
                    del img_d
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm(model.parameters(), 12)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss_ce, loss_dice, loss, logits = get_forward(img_d, lab_d, crit_ce, crit_dice, model)
                del img_d
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 12)
                optimizer.step()

            ddp_print(f'Temp: {logits[:,0].min().item(), logits[:,0].max().item(), logits[:,1].min().item(), logits[:,1].max().item()}')
            #print(logits.shape, lab.shape, lab.max(), lab.min())
            
            
            avg_loss_ce += loss_ce.item()
            avg_loss_dice += loss_dice.item()

            if it % 2 == 0:
                ddp_print(f'[{epoch}/{it}] loss_ce={loss_ce:.5f}, loss_dice={loss_dice:.5f}, time: {time.time() - t0:.4f}s')

        avg_loss_ce /= args.step_per_epoch
        avg_loss_dice /= args.step_per_epoch

        # do validation
        if epoch % args.test_frequency == 0:
            ddp_print('Test on val set')
            val_loss_ce, val_loss_dice = validate(model, val_loader, args.device, crit_ce, crit_dice, epoch, debug=debug)
            ddp_print(f'[Val{epoch}] average ce loss and dice loss are {val_loss_ce:.5f}, {val_loss_dice:.5f}')
            # save the model
            if args.is_master and val_loss_dice < best_loss_dice:
                best_loss_dice = val_loss_dice
                print(f'Saving the model at epoch {epoch} with dice loss {best_loss_dice:.4f}')
                torch.save(model, os.path.join(args.save_folder, 'best_model.pt'))
                

        # save image for subsequent analysis
        if debug and args.is_master:
            save_image_in_training(imgfiles, img, lab, logits, epoch, 'train', debug_idx)

        # learning rate decay
        cur_lr = poly_lr(epoch, args.max_epochs, args.lr, 0.9)
        ddp_print(f'Setting lr to {cur_lr}')
        for g in optimizer.param_groups:
            g['lr'] = cur_lr

def main():
    # keep track of master, useful for IO
    args.is_master = args.local_rank == 0
    # set device
    if args.cpu:
        args.device = util.init_device('cpu')
    else:
        args.device = util.init_device(args.local_rank)
    # initialize group
    distrib.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)

    train()

if __name__ == '__main__':
    main()
    






