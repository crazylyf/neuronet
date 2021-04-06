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

def init_device(device_name):
    if device_name == 'cpu':
        device = torch.device(device_name)
    elif device_name[:4] == 'cuda':
        if torch.cuda.is_available():
            device = torch.device(device_name)
    else:
        raise ValueEror("Invalid argument for device_name")

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


# TODO: network ploting
