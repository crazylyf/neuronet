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
parser.add_argument('--use_robust_loss', action='store_true', 
                    help='Whether to use robust loss')
parser.add_argument('--max_epochs', default=200, type=int,
                    help='maximal number of epochs')
parser.add_argument('--step_per_epoch', default=200, type=int,
                    help='step per epoch')
parser.add_argument('--deterministic', action='store_true',
                    help='run in deterministic mode')
parser.add_argument('--test_frequency', default=3, type=int,
                    help='frequency of testing')
parser.add_argument('--print_frequency', default=5, type=int,
                    help='frequency of information logging')
parser.add_argument('--local_rank', default=-1, type=int, metavar='N', 
                    help='Local process rank')  # DDP required
parser.add_argument('--seed', default=1025, type=int,
                    help='Random seed value')
parser.add_argument('--checkpoint', default='', type=str,
                    help='Saved checkpoint')
parser.add_argument('--evaluation', action='store_true',
                    help='evaluation')
parser.add_argument('--eval_flip', action='store_true',
                    help='whether flip image to do sample ensemble')
parser.add_argument('--lr_steps', default='40,50,60,70,80,90,95', type=str,
                    help='Steps for step_lr policy')
# network specific
parser.add_argument('--net_config', default="./models/configs/default_config.json",
                    type=str,
                    help='json file defining the network configurations')

parser.add_argument('--save_folder', default='exps/temp',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()




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

def get_fn_weights(lab_d, probs, bg_thresh=0.5, weight_fn=5.0, start_epoch=5):
    if args.curr_epoch < start_epoch:
        loss_weights, loss_weights_unsq = 1.0, 1.0
    else:
        pos_mask = lab_d > 0
        bg_mask = probs[:,0] > bg_thresh
        fn_mask = pos_mask & bg_mask
        loss_weights = torch.ones(fn_mask.size(), dtype=probs.dtype, device=probs.device)
        loss_weights[fn_mask] = weight_fn
        loss_weights_unsq = loss_weights.unsqueeze(1)
    return loss_weights, loss_weights_unsq

def get_forward(img_d, lab_d, crit_ce, crit_dice, model):
    logits = model(img_d)
    if isinstance(logits, list):
        weights = [1./2**i for i in range(len(logits))]
        sum_weights = sum(weights)
        weights = [w / sum_weights for w in weights]
    else:
        weights = [1.]
        logits = [logits]

    loss_ce_items, loss_dice_items = [], []

    for i in range(len(logits)):
        # get prediction
        probs = F.softmax(logits[i][:,:2], dim=1)
        probs0 = F.softmax(logits[i][:,2:], dim=1)

        # hard positive mining. NOTE: we can only do positive mining, as the label is incomplete
        do_hard_pos_mining = True
        if do_hard_pos_mining:
            loss_weights, loss_weights_unsq = get_fn_weights(lab_d[:,0], probs, bg_thresh=0.5, weight_fn=1.5, start_epoch=5)
            loss_weights0, loss_weights_unsq0 = get_fn_weights(lab_d[:,1], probs0, bg_thresh=0.5, weight_fn=1.5, start_epoch=5)
        else:
            loss_weights, loss_weights_unsq = 1.0, 1.0
            loss_weights0, loss_weights_unsq0 = 1.0, 1.0

        loss_ce = (crit_ce(logits[i][:,:2], lab_d[:,0].long()) * loss_weights).mean() + \
                    (crit_ce(logits[i][:,2:], lab_d[:,1].long()) * loss_weights0).mean()
        loss_dice = crit_dice(probs * loss_weights_unsq, lab_d[:,0].float() * loss_weights) + \
                    crit_dice(probs0 * loss_weights_unsq, lab_d[:,1].float() * loss_weights0)
        loss_ce_items.append(loss_ce.item())
        loss_dice_items.append(loss_dice.item())
        if i == 0:
            loss = (loss_ce + loss_dice) * weights[i]
        else:
            loss += (loss_ce + loss_dice) * weights[i]
    return loss_ce_items, loss_dice_items, loss, logits[0][:,2:]

def get_forward_eval(img_d, lab_d, crit_ce, crit_dice, model):
    if args.amp:
        with autocast():
            with torch.no_grad():
                loss_ces, loss_dices, loss, logits = get_forward(img_d, lab_d, crit_ce, crit_dice, model)
    else:
        with torch.no_grad():
            loss_ces, loss_dices, loss, logits = get_forward(img_d, lab_d, crit_ce, crit_dice, model)
    return loss_ces, loss_dices, loss, logits


def validate(model, val_loader, crit_ce, crit_dice, epoch, debug=True, num_image_save=10, phase='val'):
    model.eval()
    num_saved = 0
    if num_image_save == -1:
        num_image_save = 9999

    if phase == 'test':
        from neuronet.evaluation.multi_crop_evaluation import NonOverlapCropEvaluation, MostFitCropEvaluation
        
        noce = MostFitCropEvaluation(args.imgshape, args.ds_ratios)

        assert args.batch_size == 1, "Batch size must be 1 for test phase for current version"

    losses = []
    processed = -1
    for img,lab,imgfiles,swcfiles in val_loader:
        processed += 1
        if phase == 'test':
            ddp_print(f'==> processed: {processed}')

        img, lab = crop_data(img, lab)
        img_d = img.to(args.device)
        lab_d = lab.to(args.device)
        if phase == 'val':
            loss_ces, loss_dices, loss, logits = get_forward_eval(img_d, lab_d, crit_ce, crit_dice, model)
        elif phase == 'test':
            n_ens = 4 if args.eval_flip else 1
            for ie in range(n_ens):
                if ie == 0:
                    crops, crop_sizes, lab_crops = noce.get_image_crops(img_d[0], lab_d[0])
                elif ie == 1:
                    crops, crop_sizes, lab_crops = noce.get_image_crops(torch.flip(img_d[0], [2]), torch.flip(lab_d[0], [1]))
                elif ie == 2:
                    crops, crop_sizes, lab_crops = noce.get_image_crops(torch.flip(img_d[0], [3]), torch.flip(lab_d[0], [2]))
                elif ie == 3:
                    crops, crop_sizes, lab_crops = noce.get_image_crops(torch.flip(img_d[0], [2,3]), torch.flip(lab_d[0], [1,2]))

                logits_list = []
                loss_ces, loss_dices, loss = [], [], 0
                for i in range(len(crops)):
                    loss_ces_i, loss_dices_i, loss_i, logits_i = get_forward_eval(crops[i][None], lab_crops[i][None], crit_ce, crit_dice, model)
                    logits_list.append(logits_i[0])
                    loss_ces.append(loss_ces_i)
                    loss_dices.append(loss_dices_i)
                    loss += loss_i

                # merge the crop of prediction to unit one
                logits = noce.get_pred_from_crops(lab_d[0,0].shape, logits_list, crop_sizes)[None]
                if ie == 0:
                    avg_logits = logits
                elif ie == 1:
                    avg_logits += torch.flip(logits, [3])
                elif ie == 2:
                    avg_logits += torch.flip(logits, [4])
                elif ie == 3:
                    avg_logits += torch.flip(logits, [3,4])
                else:
                    raise ValueError

                ncrop = len(crops)
                del crops, lab_crops, logits_list

                # average the loss
                loss_ces = np.array(loss_ces).mean(axis=0)
                loss_dices = np.array(loss_dices).mean(axis=0)
                loss /= ncrop
                #TODO: the loss for ensemble mode should also averaged
                
            # averaging all logits
            logits = avg_logits / n_ens
        else:
            raise ValueError

        del img_d
        del lab_d

        losses.append([loss_ces[0], loss_dices[0], loss.item()])

        if debug:
            for debug_idx in range(img.size(0)):
                num_saved += 1
                if num_saved > num_image_save:
                    break
                save_image_in_training(imgfiles, img, lab[:,1], logits, epoch, phase, debug_idx)

    losses = torch.from_numpy(np.array(losses)).to(args.device)
    distrib.all_reduce(losses, op=distrib.ReduceOp.SUM)
    losses = losses.mean(dim=0) / distrib.get_world_size()
    
    return losses

def load_dataset(phase, imgshape):
    dset = GenericDataset(args.data_file, phase=phase, imgshape=imgshape)
    ddp_print(f'Number of {phase} samples: {len(dset)}')
    # distributedSampler
    if phase == 'train':
        sampler = DistributedSampler(dset, shuffle=True)
    else:
        sampler = DistributedSampler(dset, shuffle=False)

    loader = tudata.DataLoader(dset, args.batch_size, 
                                num_workers=args.num_workers, 
                                shuffle=False, pin_memory=True, 
                                sampler=sampler,
                                drop_last=True, 
                                worker_init_fn=util.worker_init_fn)
    dset_iter = iter(loader)
    return loader, dset_iter

def evaluate(model, optimizer, crit_ce, crit_dice, imgshape):
    phase = 'test'
    val_loader, val_iter = load_dataset(phase, imgshape)
    args.curr_epoch = 0
    loss_ce, loss_dice, loss = validate(model, val_loader, crit_ce, crit_dice, epoch=0, debug=True, num_image_save=-1, phase=phase)
    ddp_print(f'Average loss_ce and loss_dice: {loss_ce:.5f} {loss_dice:.5f}')

def train(model, optimizer, crit_ce, crit_dice, imgshape):
    # dataset preparing
    train_loader, train_iter = load_dataset('train', imgshape)
    val_loader, val_iter = load_dataset('val', imgshape)
    
    # training process
    model.train()
    
    t0 = time.time()
    grad_scaler = GradScaler()
    debug = True
    debug_idx = 0
    best_loss_dice = 1.0e10
    for epoch in range(args.max_epochs):
        # push the epoch information to global namespace args
        args.curr_epoch = epoch

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
                    loss_ces, loss_dices, loss, logits = get_forward(img_d, lab_d, crit_ce, crit_dice, model)
                    del img_d
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm(model.parameters(), 12)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss_ces, loss_dices, loss, logits = get_forward(img_d, lab_d, crit_ce, crit_dice, model)
                del img_d
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 12)
                optimizer.step()

            avg_loss_ce += loss_ces[0]
            avg_loss_dice += loss_dices[0]

            # train statistics for bebug afterward
            if it % args.print_frequency == 0:
                ddp_print(f'[{epoch}/{it}] loss_ce={loss_ces[0]:.5f}, loss_dice={loss_dices[0]:.5f}, time: {time.time() - t0:.4f}s')
                ddp_print(f'----> [{it}] Logits info: {logits[:,0].min().item():.5f}, {logits[:,0].max().item():.5f}, {logits[:,1].min().item():.5f}, {logits[:,1].max().item():.5f}')

        avg_loss_ce /= args.step_per_epoch
        avg_loss_dice /= args.step_per_epoch

        # do validation
        if epoch % args.test_frequency == 0:
            ddp_print('Evaluate on val set')
            val_loss_ce, val_loss_dice, val_loss = validate(model, val_loader, crit_ce, crit_dice, epoch, debug=debug, phase='val')
            model.train()   # back to train phase
            ddp_print(f'[Val{epoch}] average ce loss and dice loss are {val_loss_ce:.5f}, {val_loss_dice:.5f}')
            # save the model
            if args.is_master:
                # save current model
                torch.save(model, os.path.join(args.save_folder, 'final_model.pt'))

                if val_loss_dice < best_loss_dice:
                    best_loss_dice = val_loss_dice
                    print(f'Saving the model at epoch {epoch} with dice loss {best_loss_dice:.4f}')
                    torch.save(model, os.path.join(args.save_folder, 'best_model.pt'))
            

        # save image for subsequent analysis
        if debug and args.is_master and epoch % args.test_frequency == 0:
            save_image_in_training(imgfiles, img, lab[:,1], logits, epoch, 'train', debug_idx)

        # learning rate decay
        cur_lr = util.step_lr(epoch, args.lr_steps, args.lr, 0.3)
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

    if args.deterministic:
        util.set_deterministic(deterministic=True, seed=args.seed)

    # for output folder
    if args.is_master and not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    # Network
    with open(args.net_config) as fp:
        net_configs = json.load(fp)
        print('Network configs: ', net_configs)
        model = unet.UNet(**net_configs)
        ddp_print('\n' + '='*10 + 'Network Structure' + '='*10)
        ddp_print(model)
        ddp_print('='*30 + '\n')

    # get the network downsizing informations
    ds_ratios = np.array([1,1,1])
    for stride in net_configs['stride_list']:
        ds_ratios *= np.array(stride)
    args.ds_ratios = tuple(ds_ratios.tolist())

    model = model.to(args.device)
    if args.checkpoint:
        # load checkpoint
        ddp_print(f'Loading checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location={'cuda:0':f'cuda:{args.local_rank}'})
        model.load_state_dict(checkpoint.module.state_dict())
        del checkpoint
    
    # convert to distributed data parallel model
    model = DDP(model, device_ids=[args.local_rank],
                output_device=args.local_rank)#, find_unused_parameters=True)

    # optimizer & loss
    if args.checkpoint:
        args.lr /= 5
        # note: SGD is thought always better than Adam if training time
        # is long enough
        #optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay, amsgrad=True)
    else:
        #optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay, amsgrad=True)
        optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    crit_ce = nn.CrossEntropyLoss(reduction='none').to(args.device)
    crit_dice = BinaryDiceLoss(smooth=1e-5, input_logits=False).to(args.device)

    args.imgshape = tuple(map(int, args.image_shape.split(',')))
    args.lr_steps = tuple(map(int, args.lr_steps.split(',')))

    # Print out the arguments information
    ddp_print('Argument are: ')
    ddp_print(f'   {args}')
 
    if args.evaluation:
        evaluate(model, optimizer, crit_ce, crit_dice, args.imgshape)
    else:
        train(model, optimizer, crit_ce, crit_dice, args.imgshape)

if __name__ == '__main__':
    main()
    






