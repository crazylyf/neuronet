#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : image_util.py
#   Author       : Yufeng Liu
#   Date         : 2021-04-05
#   Description  : 
#
#================================================================

def normalize_normal(image4d, mask=None):
    assert image4d.ndim == 4, "image must in 4 dimension: c,z,y,x"
    assert image4d.dtype.name.startswith('float')
    for c in range(image4d.shape[0]):
        if mask is None:
            image4d[c] = (image4d[c] - image4d[c].mean()) / (image4d[c].std() + 1e-8)
        else:
            image4d[c][mask] = (image4d[c][mask] - image4d[c][mask].mean()) / (image4d[c][mask].std() + 1e-8) 
            image4d[c][mask==0] = 0 
    return image4d

def unnormalize_normal(image4d, output_range=(0,255)):
    assert image4d.ndim == 4
    assert image4d.dtype.name.startswith('float')
    for c in range(image4d.shape[0]):
        or1, or2 = output_range
        m1, m2 = image4d[c].min(), image4d[c].max()
        image4d[c] = (image4d[c] - m1) * (or2 - or1) / (m2 - m1 + 1e-8) + or1
    return image4d

