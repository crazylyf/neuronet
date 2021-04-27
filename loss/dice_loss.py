#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : dice_loss.py
#   Author       : Yufeng Liu
#   Date         : 2021-04-07
#   Description  : 
#
#================================================================

import torch
from torch import nn
import torch.nn.functional as F

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, p=1, reduction='mean', input_logits=True):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.input_logits = input_logits

    def forward(self, logits, gt):
        assert logits.shape[0] == gt.shape[0], "batch size error!"
        gt = gt.float() # int to float
        if self.input_logits:
            probs = F.softmax(logits, dim=1)[:,1]    # foreground
        else:
            probs = logits[:,1]        

        probs = probs.contiguous().view(probs.shape[0], -1)
        gt = gt.contiguous().view(gt.shape[0], -1)

        nominator = 2 * torch.sum(torch.mul(probs, gt), dim=1) + self.smooth
        if self.p == 1:
            denominator = torch.sum(probs + gt, dim=1) + self.smooth
        elif self.p == 2:
            denominator = torch.sum(probs*probs + gt*gt, dim=1) + self.smooth
        else:
            raise NotImplementedError

        loss = 1 - nominator / denominator
        return loss.mean()
        


