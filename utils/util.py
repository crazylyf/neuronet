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

import torch

def init_device(device_name):
    if device_name == 'cpu':
        device = torch.device(device_name)
    elif device_name[:4] == 'cuda':
        if torch.cuda.is_available():
            device = torch.device(device_name)
    else:
        raise ValueEror, "Invalid argument for device_name"


# TODO: network ploting
