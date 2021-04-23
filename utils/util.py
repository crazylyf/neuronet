#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : util.py
#   Author       : Yufeng Liu
#   Date         : 2021-03-26
#   Description  : 
#
#================================================================

import numpy as np
import torch
import torch.nn.functional as F

def init_device(device_name):
    if type(device_name) == int:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            raise EnvironmentError("GPU is not accessible!")
    elif type(device_name) == str:
        if device_name == 'cpu':
            device = torch.device(device_name)
        elif device_name[:4] == 'cuda':
            if torch.cuda.is_available():
                device = torch.device(device_name)
        else:
            raise ValueError("Invalid name for device")
    else:
        raise NotImplementedError
    return device

def set_deterministic(deterministic=True, seed=1024):
    if deterministic:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return True

def worker_init_fn(worker_id):
    """Function to avoid numpy.random seed duplication across multi-threads"""
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def logits_to_seg(logits, thresh=None):
    with torch.no_grad():
        if thresh is not None:
            # no need to do expensive softmax
            seg = logits.argmax(dim=1)
        else:
            probs = F.softmax(logits, dim=1)
            vmax, seg = probs.max(dim=1)
            mask = vmax > thresh
            # thresh for non-zero class
            seg[~mask] = 0
    return seg

def get_grad_stats(model):
    stats = []
    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        grad_data = p.grad.data
        norm = grad_data.norm(2).item()
        gmin = grad_data.min().item()
        gmax = grad_data.max().item()
        gmean = grad_data.mean().item()
        stats.append((norm, gmin, gmax, gmean))
    return stats

def get_param_stats(model):
    stats = []
    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        param_data = p.data
        norm = param_data.norm(2).item()
        gmin = param_data.min().item()
        gmax = param_data.max().item()
        gmean = param_data.mean().item()
        stats.append((norm, gmin, gmax, gmean))
    return stats


# TODO: network ploting
