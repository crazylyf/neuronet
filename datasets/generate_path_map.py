#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : generate_path_map.py
#   Author       : Yufeng Liu
#   Date         : 2021-04-15
#   Description  : 
#
#================================================================

import sys
import numpy as np
from skimage.draw import line_nd
from skimage.io import imsave

from swc_handler import parse_swc

sys.setrecursionlimit(30000)
#counter = 0

def get_kernel(maxv, intercept=2, fn='cubic', z_space=3.0):
    """
    
    """
    if fn == 'cubic':
        radius = int(maxv**(1/3)) + 1
        # grid coordinates
        kx = ky = radius
        kz = int(radius / z_space) + 1
        mgz,mgy,mgx = np.meshgrid(range(-kz,kz+1), range(-ky,ky+1), range(-kx,kx+1))
        svalues = (z_space*np.fabs(mgz))**3 + np.fabs(mgy)**3 + np.fabs(mgx)**3 + intercept
        svalues[svalues > maxv] = maxv
        svalues.flags.writeable = False
        return svalues, (mgz,mgy,mgx)
    else:
        raise NotImplementedError

def assign_neighbouring_values(img, pos, kernel):
    k = (np.array(kernel.shape) - 1) // 2
    pos = np.array(pos)
    s = np.maximum(pos - k, 0)
    e = np.minimum(pos + k + 1, img.shape)

    ks = np.maximum(k - pos, 0)
    ke = ks + e - s
    img[s[0]:e[0], s[1]:e[1], s[2]:e[2]] = np.minimum(kernel[ks[0]:ke[0], ks[1]:ke[1], ks[2]:ke[2]] + img[pos[0],pos[1],pos[2]], img[s[0]:e[0], s[1]:e[1], s[2]:e[2]])

def is_in_box(x, y, z, imgshape):
    """ 
    imgshape must be in (z,y,x) order
    """
    if x < 0 or y < 0 or z < 0 or \
        x > imgshape[2] - 1 or \
        y > imgshape[1] - 1 or \
        z > imgshape[0] - 1:
        return False
    return True

def dfs_traverse(idx, child_dict, pos_dict, img, kernel):
    '''
    Assuming that most points are within current volume
    '''
    leaf = pos_dict[idx]
    pos = np.array(leaf[2:5][::-1]) # to z-y-x order
    p_idx = leaf[-1]    # parent index
    #global counter
    #counter += 1
    #print(counter)
    
    if p_idx == -1: # soma
        img[pos[0],pos[1],pos[2]] = 0
        assign_neighbouring_values(img, pos, kernel)
    else:
        p_pos = pos_dict[p_idx][2:5][::-1]
        
        # set path value for each point
        # firstly, get all points between current point and its parent
        lin = line_nd(p_pos, pos, endpoint=True)
        #print(p_pos, pos)
        #print(f'==> {len(lin[0])}')
        
        for (i,z,y,x) in zip(range(len(lin[0])),*lin):
            if i == 0:
                # the parent point, already assigned
                p_pos = [z,y,x]
            elif not is_in_box(x,y,z, img.shape):
                break
            else:
                img[z,y,x] = min(img[p_pos[0],p_pos[1],p_pos[2]] + 1, img[z,y,x])
                assign_neighbouring_values(img, [z,y,x], kernel)

    # stop criterion
    if idx not in child_dict:
        return 

    for new_idx in child_dict[idx]:
        dfs_traverse(new_idx, child_dict, pos_dict, img, kernel)
    
    
def find_soma_node(tree, p_soma=-1):
    for leaf in tree:
        if leaf[-1] == p_soma:
            #print('Soma: ', leaf)
            return leaf[0]
    raise ValueError("Could not find the soma node!")

def generate_path_map(tree, imgshape, maxv, kernel, intercept=2, fn='cubic', z_space=3.0, normalize=True):
    '''
    imgshape in order (z,y,x)
    '''
    # initialize empty image
    img = np.ones(imgshape, dtype=np.float32) * maxv
    # initialize position dict
    pos_dict = {}
    for i, leaf in enumerate(tree):
        idx, type_, x, y, z, r, p = leaf
        x = int(round(x))
        y = int(round(y))
        z = int(round(z))
        pos_dict[leaf[0]] = (idx, type_, x, y, z, r, p)
    soma_idx = find_soma_node(tree, p_soma=-1)
    # initialize parent-child pairs
    child_dict = {}
    for leaf in tree:
        if leaf[-1] in child_dict:
            child_dict[leaf[-1]].append(leaf[0])
        else:
            child_dict[leaf[-1]] = [leaf[0]]

    
    # do dfs traverse
    dfs_traverse(soma_idx, child_dict, pos_dict, img, kernel)

    if normalize:
        img /= maxv
    # flip the image in y-axis
    img = img[:,::-1]
    
    return img


if __name__ == '__main__':
    import time

    swcfile = '9876_5060_2782_trim.swc'
    imgshape = (256,512,512)
    maxv = 1600

    tree = parse_swc(swcfile)
    kernel, _ = get_kernel(maxv, intercept=intercept, fn=fn, z_space=z_space)
    print("Kernel: ", kernel.min(), kernel.max())
    t0 = time.time()
    img = generate_path_map(tree, imgshape, maxv, kernel, intercept=2, fn='cubic', z_space=3.0)
    print(f'Map generateion take {time.time() - t0}s')
    print(img.mean(), img.std(), img.max(), img.min())
    #import ipdb; ipdb.set_trace()
    img = 255 - (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)
    
    imsave('test.tiff', img)


