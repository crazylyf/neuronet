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

def random_crop_3D_image(img, crop_size):
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * len(img.shape)
    else:
        assert len(crop_size) == len(
            img.shape), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"

    if crop_size[0] < img.shape[0]:
        lb_x = np.random.randint(0, img.shape[0] - crop_size[0])
    elif crop_size[0] == img.shape[0]:
        lb_x = 0
    else:
        raise ValueError("crop_size[0] must be smaller or equal to the images x dimension")

    if crop_size[1] < img.shape[1]:
        lb_y = np.random.randint(0, img.shape[1] - crop_size[1])
    elif crop_size[1] == img.shape[1]:
        lb_y = 0
    else:
        raise ValueError("crop_size[1] must be smaller or equal to the images y dimension")

    if crop_size[2] < img.shape[2]:
        lb_z = np.random.randint(0, img.shape[2] - crop_size[2])
    elif crop_size[2] == img.shape[2]:
        lb_z = 0
    else:
        raise ValueError("crop_size[2] must be smaller or equal to the images z dimension")

    return img[lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]], lb_x, lb_y, lb_z

