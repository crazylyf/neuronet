#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : preprocess.py
#   Author       : Yufeng Liu
#   Date         : 2021-03-31
#   Description  : This package tries to standardize the input image, 
#                  for lowerize the burden when training, including: 
#                  - resampling
#                  - normalization
#                  - format conversion
#                  - dataset splitting
#                  
#================================================================

import os, glob
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
from copy import deepcopy
import SimpleITK as sitk
from multiprocessing.pool import Pool
import pickle
from skimage.draw import line_nd
import skimage.morphology as morphology

from swc_handler import parse_swc, write_swc
from path_util import get_file_prefix


def soma_labelling(image, z_ratio=0.3, r=9, thresh=220, label=255):
    dz, dy, dx = image.shape

    img_thresh = image.copy()
    img_thresh[img_thresh > thresh] = thresh
    
    cx, cy, cz = dx//2, dy//2, dz//2
    rz = int(round(z_ratio * r)) 
    img_labels = []
    zs, ze = cz - rz, cz + rz
    ys, ye = cy - r, cy + r 
    xs, xe = cx - r, cx + r 
    img_thresh[zs:ze, ys:ye, xs:xe] = label
    
    return img_thresh

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

def trim_swc(tree_orig, imgshape, keep_candidate_points=True):
    """
    Trim the out-of-box and non_connecting leaves
    """

    def traverse_leaves(idx, child_dict, good_points, cand_pints, pos_dict):
        leaf = pos_dict[idx]
        p_idx, ib = leaf[-2:]

        if (p_idx in good_points) or (p_idx == -1):
            if ib: 
                good_points.add(idx)    # current node
            else:
                cand_points.add(idx)
                return

        if idx not in child_dict:
            return

        for new_idx in child_dict[idx]:
            traverse_leaves(new_idx, child_dict, good_points, cand_pints, pos_dict) 


    # execute trimming
    pos_dict = {}
    tree = deepcopy(tree_orig)
    for i, leaf in enumerate(tree_orig):
        idx, type_, x, y, z, r, p = leaf
        leaf = (idx, type_, x, y, z, r, p, is_in_box(x,y,z,imgshape))
        pos_dict[idx] = leaf
        tree[i] = leaf

    good_points = set() # points and all its upstream parents are in-box
    cand_points = set() # all upstream parents are in-box, itself not
    # initialize the visited set with soma, whose parent index is -1
    soma_idx = None
    for leaf in tree:
        if leaf[-2] == -1:
            soma_idx = leaf[0]
            break

    child_dict = {}
    for leaf in tree:
        if leaf[-2] in child_dict:
            child_dict[leaf[-2]].append(leaf[0])
        else:
            child_dict[leaf[-2]] = [leaf[0]]
    # do DFS searching
    #print(soma_idx)
    traverse_leaves(soma_idx, child_dict, good_points, cand_points, pos_dict)
    #print("#good/#cand/#total:", len(good_points), len(cand_points), len(pos_dict))  
    
    # return the tree, (NOTE: without order)
    tree_trim = []
    if keep_candidate_points:
        keep_points = good_points | cand_points
    else:
        keep_points = good_points

    for i, leaf in enumerate(tree):
        idx = leaf[0]
        if idx in keep_points:
            tree_trim.append(leaf[:-1])

    return tree_trim

def trim_out_of_box(tree_orig, imgshape, keep_candidate_points=True):
    """
    Trim the out-of-box leaves
    """
    # execute trimming
    child_dict = {}
    for leaf in tree_orig:
        if leaf[-1] in child_dict:
            child_dict[leaf[-1]].append(leaf[0])
        else:
            child_dict[leaf[-1]] = [leaf[0]]
    
    pos_dict = {}
    for i, leaf in enumerate(tree_orig):
        pos_dict[leaf[0]] = leaf

    tree = []
    for i, leaf in enumerate(tree_orig):
        idx, type_, x, y, z, r, p = leaf
        ib = is_in_box(x,y,z,imgshape)
        if ib:
            tree.append(leaf)
        elif keep_candidate_points:
            if p in pos_dict and is_in_box(*pos_dict[p][2:5], imgshape):
                tree.append(leaf)
            elif idx in child_dict:
                for ch_leaf in child_dict[idx]:
                    if is_in_box(*pos_dict[ch_leaf][2:5], imgshape):
                        tree.append(leaf)
    return tree

def load_spacing(spacing_file, zyx_order=True):
    spacing_dict = {}
    with open(spacing_file) as fp:
        fp.readline()    # skip the first line
        for line in fp.readlines():
            line = line.strip()
            if not line: continue

            brain_id, xs, ys, zs = line.split(',')
            if zyx_order:
                spacing_dict[int(brain_id)] = np.array([float(zs), float(ys), float(xs)])
            else:
                spacing_dict[int(brain_id)] = np.array([float(xs), float(ys), float(zs)])
    
    return spacing_dict

def swc_to_image(tree, r_exp=3, z_ratio=0.4, imgshape=(256,512,512), flipy=True, label_soma=False):
    # Note imgshape in (z,y,x) order
    # initialize empty image
    img = np.zeros(shape=imgshape, dtype=np.uint8)
    # get the position tree and parent tree
    pos_dict = {}
    for i, leaf in enumerate(tree):
        idx, type_, x, y, z, r, p = leaf
        leaf = (idx, type_, x, y, z, r, p, is_in_box(x,y,z,imgshape))
        pos_dict[idx] = leaf
        tree[i] = leaf
        
    xl, yl, zl = [], [], []
    for _, leaf in pos_dict.items():
        idx, type_, x, y, z, r, p, ib = leaf
        if p < 1: continue   # soma
        
        parent_leaf = pos_dict[p]
        if (not ib) and (not parent_leaf[ib]):
            print('All points are out of box! do trim_swc before!')
            raise ValueError
        
        # draw line connect each pair
        cur_pos = leaf[2:5]
        par_pos = parent_leaf[2:5]
        lin = line_nd(cur_pos[::-1], par_pos[::-1], endpoint=True)

        xl.extend(list(lin[2]))
        yl.extend(list(lin[1]))
        zl.extend(list(lin[0]))

    xn, yn, zn = [], [], []
    for (xi,yi,zi) in zip(xl,yl,zl):
        if is_in_box(xi,yi,zi,imgshape):
            xn.append(xi)
            yn.append(yi)
            zn.append(zi)
    img[zn,yn,xn] = 1

    # do morphology expansion
    r_z = max(int(round(z_ratio * r_exp)), 1)
    selem = np.ones((r_z, r_exp, r_exp), dtype=np.uint8)
    img = morphology.dilation(img, selem)
    # soma-labelling
    if label_soma:
        lab_img = soma_labelling(img, r=r_exp*2+1, thresh=220, label=1)
    else:
        lab_img = img
    
    if flipy:
        lab_img = lab_img[:,::-1]   # flip in y-axis, as the image is flipped

    return lab_img

def swc_to_fullconnect(tree):
    # get the position tree and parent tree
    pos_dict = {}
    for i, leaf in enumerate(tree):
        idx, type_, x, y, z, r, p = leaf
        # manually convert to integar, in case of rounding problems
        x = int(round(x))
        y = int(round(y))
        z = int(round(z))
        pos_dict[leaf[0]] = (idx, type_, x, y, z, r, p)
     
    fc_dict = {}
    fc_indices = {}
    xl, yl, zl = [], [], []
    for _, leaf in pos_dict.items():
        idx, type_, x, y, z, r, p = leaf

        # soma
        if p <= 0: 
            key = (int(round(x)), int(round(y)), int(round(z)))
            if key not in fc_indices:
                iid = len(fc_indices)
                fc_indices[key] = iid
                fc_dict[iid] = (iid, type_, x, y, z, r, p)
            continue
        
        parent_leaf = pos_dict[p]
        # draw line connect each pair
        cur_pos = leaf[2:5]
        par_pos = parent_leaf[2:5]
        lin = line_nd(par_pos[::-1], cur_pos[::-1], endpoint=True)
        for pos in zip(lin[2],lin[1],lin[0]):
            if pos not in fc_indices:
                iid = len(fc_indices)
                fc_indices[pos] = iid
        for i in range(1, len(lin[0])):
            pos = lin[2][i], lin[1][i], lin[0][i]
            pos_pre = lin[2][i-1], lin[1][i-1], lin[0][i-1]
            iid = fc_indices[pos]
            iid_pre = fc_indices[pos_pre]
            type_ = leaf[1]
            r = leaf[5]
            fc_dict[iid] = (iid,type_,pos[0],pos[1],pos[2],r,iid_pre)
            if iid_pre not in fc_dict:
                fc_dict[iid_pre] = (iid_pre,type_,pos_pre[0],pos_pre[1],pos_pre[2],r,-2)

    assert(len(fc_dict) == len(fc_indices))
        
    return fc_dict, fc_indices

def swc_to_connection(tree, r_xy=3, r_z=1, imgshape=(256,512,512), flipy=True):
    zshape, yshape, xshape = imgshape
    assert zshape % r_z == 0
    assert yshape % r_xy == 0
    assert xshape % r_xy == 0
    zn, yn, xn = zshape//r_z, yshape//r_xy, xshape//r_xy

    fc_dict, fc_indices = swc_to_fullconnect(tree)
    #print(f'FC size: {len(fc_dict)}, {len(fc_indices)}')
    # initialize connection label
    lab = np.zeros((26,zn,yn,xn), dtype=np.uint8)
    mask = np.zeros((zn,yn,xn), dtype=np.bool)
    for iid, leaf in fc_dict.items():
        _, type_, xi, yi, zi, r, p_iid = leaf
        xx = xi // r_xy
        yy = yi // r_xy
        zz = zi // r_z
        if not is_in_box(xi,yi,zi,imgshape):
            continue
        mask[zz,yy,xx] = True
        if p_iid != -1 and p_iid != -2:
            xxi, yyi, zzi = fc_dict[p_iid][2:5]
            xxp, yyp, zzp = xxi//r_xy, yyi//r_xy, zzi//r_z
            if not is_in_box(xxi,yyi,zzi,imgshape):
                continue
            
            class_id = 9 * (xxp - xx + 1) + 3 * (yyp - yy + 1) + (zzp - zz + 1) - 1
            if class_id == -1:
                continue
            else:
                lab[class_id,zz,yy,xx] = 1
                lab[25-class_id,zzp,yyp,xxp] = 1
    #mask = lab.sum(axis=0) > 0
    if flipy:
        mask = mask[:,::-1]
        lab = lab[:,:,::-1]

    return mask, lab

def check_connectivity(mask):
    indices = np.nonzero(mask)
    for zz,yy,xx in zip(*indices):
        zs = max(0, zz-1)
        ze = min(mask.shape[0]-1, zz+1)
        ys = max(0, yy-1)
        ye = min(mask.shape[1]-1, yy+1)
        xs = max(0, xx-1)
        xe = min(mask.shape[2]-1, xx+1)
        neighbouring = mask[zs:ze+1, ys:ye+1, xs:xe+1]
        if neighbouring.sum() < 0:
            return False
    return True
    
    
if __name__ == '__main__':
    import time

    prefix = '8315_19523_2299'
    swc_file = f'/home/lyf/Research/auto_trace/neuronet/data/task0001_17302/{prefix}.swc'
    imgshape = (256,384,384)
    
    t0 = time.time()    
    tree = parse_swc(swc_file)
    print(f'Parsing: {time.time() - t0:.4f}s')
    tree = trim_swc(tree, imgshape)
    print(f'Trim: {time.time() - t0:.4f}s')
    mask, lab = swc_to_connection(tree, r_xy=3, r_z=1, imgshape=imgshape, flipy=True)
    valid = check_connectivity(mask)
    print(f'Validicity: {valid}')
    print(f'Labelling: {time.time() - t0:.4f}s')
    import ipdb; ipdb.set_trace()
    sitk.WriteImage(sitk.GetImageFromArray(lab.max(axis=0).astype(np.uint8)), f'{prefix}_label.tiff')
    sitk.WriteImage(sitk.GetImageFromArray(mask.astype(np.uint8)), f'{prefix}_mask.tiff')

