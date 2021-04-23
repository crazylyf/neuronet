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
        lab_v = (unnormalize_normal(lab[idx].numpy().astype(np.float32)[None])[0]).astype(np.uint8)
        
        log_v = F.softmax(logits[idx][:2], dim=0)
        log_v = log_v.cpu().numpy()[[1]].astype(np.float32)
        log_v = (unnormalize_normal(log_v)[0]).astype(np.uint8)
        sitk.WriteImage(sitk.GetImageFromArray(img_v), os.path.join(args.save_folder, f'debug_epoch{epoch}_{prefix}_{phase}_img.tiff'))
        sitk.WriteImage(sitk.GetImageFromArray(lab_v), os.path.join(args.save_folder, f'debug_epoch{epoch}_{prefix}_{phase}_lab.tiff'))
        sitk.WriteImage(sitk.GetImageFromArray(log_v), os.path.join(args.save_folder, f'debug_epoch{epoch}_{prefix}_{phase}_pred.tiff'))


def validate(model, val_loader, device, crit, crit_ce, crit_dice, epoch, debug=True, debug_idx=0):
    model.eval()
    avg_ce_loss = 0
    avg_dice_loss = 0
    avg_loss = 0
    for img,lab,mask,imgfiles,swcfiles in val_loader:
        img, lab = crop_data(img, lab)
        #mask = mask.unsqueeze(1)#.float()
        img_d = img.to(device)
        lab_d = lab.to(device)
        mask_d = mask.to(device)
        
        with torch.no_grad():
            logits = model(img_d)
            del img_d
            #loss = crit(logits * mask_d, lab_d * mask_d)
            loss_ce = crit_ce(logits[:,:2], mask_d.long())
            loss_dice = crit_dice(logits[:2], mask_d.float())
            loss = loss_ce + loss_dice

        avg_ce_loss += loss_ce.item()
        avg_dice_loss += loss_dice.item()
        avg_loss += loss.item()

    avg_ce_loss /= len(val_loader)
    avg_dice_loss /= len(val_loader)
    avg_loss /= len(val_loader)
    
    if debug and args.is_master:
        save_image_in_training(imgfiles, img, mask, logits, epoch, 'val', debug_idx)

    model.train()

    return avg_loss

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
        model = unet.UNetWithPreLayer(net_configs)
        ddp_print('\n' + '='*10 + 'Network Structure' + '='*10)
        ddp_print(model)
        ddp_print('='*30 + '\n')

    
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
        #optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
        args.lr /= 10.

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay, amsgrad=True)
    crit_ce = nn.CrossEntropyLoss().to(args.device)
    crit_dice = BinaryDiceLoss(smooth=1e-5).to(args.device)
    crit = nn.BCEWithLogitsLoss().to(args.device)

    # training process
    model.train()
    
    t0 = time.time()
    grad_scaler = GradScaler()
    debug = True
    debug_idx = 0
    best_loss = 1.0e10
    for epoch in range(args.max_epochs):
        avg_loss_ce = 0
        avg_loss_dice = 0
        avg_loss_bce = 0
        avg_loss = 0
        for it in range(args.step_per_epoch):
            try:
                img, lab, mask, imgfiles, swcfiles = next(train_iter)
            except StopIteration:
                # let all processes sync up before starting with a new epoch of training
                distrib.barrier()
                # reset the random seed, to avoid np.random & dataloader problem
                np.random.seed(args.seed + epoch)
                
                train_iter = iter(train_loader)
                img, lab, mask, imgfiles, swcfiles = next(train_iter)

            # center croping for debug, 64x128x128 patch
            img, lab = crop_data(img, lab)
            #mask = mask.unsqueeze(1)#.float()

            img_d = img.to(args.device)
            lab_d = lab.to(args.device)
            mask_d = mask.to(args.device)
            
            optimizer.zero_grad()
            if args.amp:
                with autocast():
                    logits = model(img_d)
                    del img_d
                    loss_ce = crit_ce(logits[:,:2], mask_d.long())
                    loss_dice = crit_dice(logits[:2], mask_d.float())
                    loss = loss_ce + loss_dice
                    #mask_logits = logits# * mask_d
                    #mask_lab = lab_d# * mask_d
                    #loss = crit(mask_logits, mask_lab)
                    #ddp_print(f'Logits: {mask_logits.mean().item()}, {mask_logits.max().item()}, {mask_logits.min().item()}')
                    #ddp_print(f'Lab: {mask_lab.mean().item()}, {mask_lab.max().item()}, {mask_lab.min().item()}')
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm(model.parameters(), 12)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                logits = model(img_d)
                del img_d
                loss_ce = crit_ce(logits[:,:2], mask_d.long())
                loss_dice = crit_dice(logits[:2], mask_d.float())
                loss = loss_ce + loss_dice
                #loss = crit(logits * mask_d, lab_d * mask_d)
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 12)
                optimizer.step()

            ddp_print(f'Temp: {logits[:,0].min().item(), logits[:,0].max().item(), logits[:,1].min().item(), logits[:,1].max().item()}')
            
            if False and args.is_master:
                print(f'Grad [{epoch}/{it}] ', )
                grad_stats = util.get_param_stats(model)
                for istat, stat in enumerate(grad_stats):
                    print(istat, *stat)
            
            avg_loss_ce += loss_ce.item()
            avg_loss_dice += loss_dice.item()
            avg_loss += loss.item()

            if it % 2 == 0:
                #ddp_print(f'[{epoch}/{it}] loss={loss.item():.5f}, time: {time.time() - t0:.4f}s')
                ddp_print(f'[{epoch}/{it}] loss={loss.item():.5f}, loss_ce={loss_ce.item():.5f}, loss_dice={loss_dice.item():.5f}, time: {time.time() - t0:.4f}s')

        avg_loss_ce /= args.step_per_epoch
        avg_loss_dice /= args.step_per_epoch
        avg_loss /= args.step_per_epoch

        # do validation
        if epoch % args.test_frequency == 0 and epoch != 0:
            ddp_print('Test on val set')
            val_loss = validate(model, val_loader, args.device, crit, crit_ce, crit_dice, epoch, debug=debug, debug_idx=debug_idx)
            ddp_print(f'[Val{epoch}] average loss are {val_loss:.5f}')
            # save the model
            if args.is_master and val_loss < best_loss:
                best_loss = val_loss
                print(f'Saving the model at epoch {epoch} with loss {best_loss:.4f}')
                torch.save(model, os.path.join(args.save_folder, 'best_model.pt'))
                

        # save image for subsequent analysis
        if debug and args.is_master:
            save_image_in_training(imgfiles, img, mask, logits, epoch, 'train', debug_idx)

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
    






