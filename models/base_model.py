#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : base_model.py
#   Author       : Yufeng Liu
#   Date         : 2021-03-26
#   Description  : 
#
#================================================================

import torch.nn as nn

class BaseModel(nn.Module):

    def __init__(self, ):
        super(BaseModel, self).__init__()

    def init_weights(self, ):
        pass

    def get_device(self):
        if next(self.parameters()).device == 'cpu':
            return "cpu"
        else:
            return next(self.parameters()).device.index

    def set_device(self, device):
        if device == 'cpu':
            self.cpu()
        else:
            self.cuda(device)

    def forward(self, x):
        raise NotImplementedError

if __name__ == '__main__':
    bm = BaseModel()
    print(bm)
