#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : evaluation.py
#   Author       : Yufeng Liu
#   Date         : 2021-05-11
#   Description  : 
#
#================================================================

import os, sys, glob
import numpy as np
import subprocess
from skimage.draw import line_nd
from scipy.spatial import distance_matrix

from swc_handler import parse_swc, write_swc, is_in_box

def tree_to_voxels(tree, crop_box):
    # initialize position dict
    pos_dict = {}
    new_tree = []
    for i, leaf in enumerate(tree):
        idx, type_, x, y, z, r, p = leaf
        leaf_new = (*leaf, is_in_box(x,y,z,crop_box))
        pos_dict[leaf[0]] = leaf_new
        new_tree.append(leaf_new)
    tree = new_tree

    xl, yl, zl = [], [], []
    for _, leaf in pos_dict.items():
        idx, type_, x, y, z, r, p, ib = leaf
        if p == -1: continue # soma
        
        if p not in pos_dict:
            continue

        parent_leaf = pos_dict[p]
        if (not ib) and (not parent_leaf[ib]):
            print('All points are out of box! do trim_swc before!')
            raise ValueError

        # draw line connecting each pair
        cur_pos = leaf[2:5]
        par_pos = parent_leaf[2:5]
        lin = line_nd(cur_pos[::-1], par_pos[::-1], endpoint=True)
        xl.extend(list(lin[2]))
        yl.extend(list(lin[1]))
        zl.extend(list(lin[0]))

    voxels = []
    for (xi,yi,zi) in zip(xl,yl,zl):
        if is_in_box(xi,yi,zi,crop_box):
            voxels.append((xi,yi,zi))
    # remove duplicate points
    voxels = np.array(list(set(voxels)), dtype=np.float32)
    return voxels
     

def get_specific_neurite(tree, type_id):
    if (not isinstance(type_id, list)) and (not isinstance(type_id, tuple)):
        type_id = (type_id,)
    
    new_tree = []
    for leaf in tree:
        if leaf[1] in type_id:
            new_tree.append(leaf)
    return new_tree

class DistanceEvaluation(object):
    def __init__(self, crop_box, neurite_type='all'):
        self.crop_box = crop_box
        self.neurite_type = neurite_type

    def calc_ESA(self, voxels1, voxels2, dist_type):
        if len(voxels1) > 200000 or len(voxels2) > 200000:
            if dist_type in ('ESA', 'DSA'):
                return 0, 198., 99.0
            elif dist_type == 'PDS':
                return 0.0, 1.0, 0.5
        elif len(voxels1) == 0:
            return None
            #if dist_type in ('ESA', 'DSA'):
            #    return 0., 198., 99.
            #elif dist_type == 'PDS':
            #    return 0., 1., .5
        elif len(voxels2) == 0:
            return None
            #if dist_type in ('ESA', 'DSA'):
            #    return 198., 0., 99.
            #elif dist_type == 'PDS':
            #    return 1., 0., .5

        pdist = distance_matrix(voxels1, voxels2)
        dists1 = pdist.min(axis=1)
        dists2 = pdist.min(axis=0)
        if dist_type == 'DSA':
            dists1 = dists1[dists1 > 2.0]
            dists2 = dists2[dists2 > 2.0]
            if dists1.shape[0] == 0:
                dists1 = np.array([0.])
            if dists2.shape[0] == 0:
                dists2 = np.array([0.])
        elif dist_type == 'PDS':
            dists1 = (dists1 > 2.0).astype(np.float32)
            dists2 = (dists2 > 2.0).astype(np.float32)
        print(f'Distance shape: {dists1.shape}, {dists2.shape}')
        esa1 = dists1.mean()
        esa2 = dists2.mean()
        esa = (esa1 + esa2) / 2.
        return esa1, esa2, esa

    def calc_DIADEM(self, swc_file1, swc_file2, jar_path='/home/lyf/Softwares/packages/Diadem/DiademMetric.jar'):
        exec_str = f'java -jar {jar_path} -G {swc_file1} -T {swc_file2} -x 6 -R 3 -z 2 --xyPathThresh 0.08 --zPathThresh 0.20 --excess-nodes false'
        #print(exec_str)
        output = subprocess.check_output(exec_str, shell=True)
        #print(output)
        score1 = float(output.split()[-1])

        exec_str = f'java -jar {jar_path} -G {swc_file2} -T {swc_file1} -x 6 -R 3 -z 2 --xyPathThresh 0.08 --zPathThresh 0.20 --excess-nodes false -r 17'
        output = subprocess.check_output(exec_str, shell=True)
        print(output)
        score2 = float(output.split()[-1])

        score = (score1 + score2) / 2.
        return score1, score2, score
        

    def calc_distance(self, swc_file1, swc_file2, dist_type='ESA'):
        if dist_type in ('ESA', 'DSA', 'PDS'):
            tree1 = parse_swc(swc_file1)
            tree2 = parse_swc(swc_file2)
            print(f'Length of nodes in tree1 and tree2: {len(tree1)}, {len(tree2)}')
            if self.neurite_type == 'all':
                pass
            elif self.neurite_type == 'dendrite':
                type_id = (3,4)
                tree1 = get_specific_neurite(tree1, type_id)
            elif self.neurite_type == 'axon':
                type_id = 2
                tree1 = get_specific_neurite(tree1, type_id)
            else:
                raise NotImplementedError

            # to successive voxels
            voxels1 = tree_to_voxels(tree1, self.crop_box)
            voxels2 = tree_to_voxels(tree2, self.crop_box)
            dist = self.calc_ESA(voxels1, voxels2, dist_type=dist_type)
        elif dist_type == 'DIADEM':
            dist = calc_DIADEM(swc_file1, swc_file2)
        else:
            raise NotImplementedError

        return dist

    def calc_distance_triple(self, swc_gt, swc_cmp1, swc_cmp2, dist_type='ESA'):
        if dist_type in ('ESA', 'DSA', 'PDS'):
            tree_gt = parse_swc(swc_gt)
            tree_cmp1 = parse_swc(swc_cmp1)
            tree_cmp2 = parse_swc(swc_cmp2)
            print(f'Length of nodes for gt, cmp1 and cmp2: {len(tree_gt)}, {len(tree_cmp1)}, {len(tree_cmp2)}')
            if self.neurite_type == 'all':
                pass
            elif self.neurite_type == 'dendrite':
                type_id = (3,4)
                tree_gt = get_specific_neurite(tree_gt, type_id)
            elif self.neurite_type == 'axon':
                type_id = 2
                tree_gt = get_specific_neurite(tree_gt, type_id)
            else:
                raise NotImplementedError

            # to successive voxels
            voxels_gt = tree_to_voxels(tree_gt, self.crop_box).astype(np.float32)
            voxels_cmp1 = tree_to_voxels(tree_cmp1, self.crop_box).astype(np.float32)
            voxels_cmp2 = tree_to_voxels(tree_cmp2, self.crop_box).astype(np.float32)
            dist1 = self.calc_ESA(voxels_gt, voxels_cmp1, dist_type=dist_type)
            dist2 = self.calc_ESA(voxels_gt, voxels_cmp2, dist_type=dist_type)
        elif dist_type == 'DIADEM':
            dist1 = self.calc_DIADEM(swc_gt, swc_cmp1)
            dist2 = self.calc_DIADEM(swc_gt, swc_cmp2)
        else:
            raise NotImplementedError
        
        return dist1, dist2

if __name__ == '__main__':
    from fuse_image_with_seg import get_image_brain_mapping

    crop_box = (256,512,512)
    base_method = 'smartTrace'
    gt_dir = f'../../data/additional_crops/swc_cropped'
    app2_dir = '../../data/additional_crops/smartTrace'
    #pred_dir = f'/home/lyf/Temp/exp040/vanilla/fused_tg0.0_alpha0.8_vanilla_bgMask0'
    pred_dir = f'../exp048_brains2/fused_tg0.0_alpha0.8_vanilla_bgMask0/smartTrace'


    mapper, img_dict = get_image_brain_mapping()
    for dist_type in ['ESA', 'DSA', 'PDS']:
        for neurite_type in ['all', 'dendrite', 'axon']:
            print(f'dist_type: {dist_type} and neurite_type: {neurite_type}')
            dists_app2 = []
            dists_pred = []
            n_success_app2 = 0
            n_success_pred = 0
            deval = DistanceEvaluation(crop_box, neurite_type=neurite_type)
            
            if base_method == 'app2':
                match_str = '[1-9]*[0-9].swc'
            elif base_method == 'smartTrace':
                match_str = '[1-9]*_smartTracing.swc'
            for swc_pred in glob.glob(os.path.join(pred_dir, match_str)):
                swc_name = os.path.split(swc_pred)[-1][:-4]
                prefix = '_'.join(swc_name.split('_')[:3]).split('.')[0]
                print(f'----- {swc_name}-------')
                brain_id = mapper[prefix]
                
                swc_gt = os.path.join(gt_dir, str(brain_id), f'{prefix}.swc')
                if base_method == 'app2':
                    swc_app2 = os.path.join(app2_dir, str(brain_id), f'{prefix}_{brain_id}.swc')
                    swc_pred = os.path.join(pred_dir, f'{prefix}_{brain_id}.swc')
                elif base_method == 'smartTrace':
                    swc_app2 = os.path.join(app2_dir, str(brain_id), f'{prefix}.tiff_smartTracing.swc')
                    swc_pred = os.path.join(pred_dir, f'{prefix}.tiff_smartTracing.swc')
                #print(os.path.exists(swc_gt), os.path.exists(swc_app2), os.path.exists(swc_pred))
                #sys.exit()

                try:
                    dist_app2, dist_pred = deval.calc_distance_triple(swc_gt, swc_app2, swc_pred, dist_type=dist_type)
                    #dist_app2 = 0
                    #dist_pred = deval.calc_distance(swc_gt, swc_pred, dist_type=dist_type)
                except:
                    continue

                print(f'{dist_type} distance between app2 and gt: {dist_app2}')
                print(f'{dist_type} distance between pred and gt: {dist_pred}')
                print(f'\n')
            
                if dist_app2 is not None:
                    n_success_app2 += 1

                if dist_pred is not None:
                    n_success_pred += 1
                
                if (dist_app2 is not None) and (dist_pred is not None):
                    dists_app2.append(dist_app2)
                    dists_pred.append(dist_pred)

            dists_app2 = np.array(dists_app2)
            dists_pred = np.array(dists_pred)
            print(f'Succeed number are: app2: {n_success_app2}, pred: {n_success_pred}\n')
            print(f'Statistics for app2: ')
            print(f'mean: ')
            print(f'    {dists_app2.mean(axis=0)}')
            print(f'std: ')
            print(f'    {dists_app2.std(axis=0)}\n')

            print(f'Statistics for pred: ')
            print(f'mean: ')
            print(f'    {dists_pred.mean(axis=0)}')
            print(f'std: ')
            print(f'    {dists_pred.std(axis=0)}')
   
