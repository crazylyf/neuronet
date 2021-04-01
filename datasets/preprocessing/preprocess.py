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
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from multiprocessing.pool import Pool

from swc_handler import parse_swc, write_swc
from path_util import get_file_prefix

def soma_labelling(image, z_ratio=0.4, r=7, thresh=220, label=255):
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

def trim_swc(tree_orig, imgshape, keep_candiate_points=True):
    """
    Trim the out-of-box and non_connecting leaves
    """
   
    def is_in_box(x, y, z, imgshape):
        if x < 0 or y < 0 or z < 0 or \
            x > imgshape[0] - 1 or \
            y > imgshape[1] - 1 or \
            z > imgshape[2] - 1:
            return False
        return True

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
    print("#good/#cand/#total:", len(good_points), len(cand_points), len(pos_dict))  
    
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

def load_data(data_dir, spacing_file, is_train=True):
    '''data_dir = '/home/lyf/data/seu_mouse/crop_data/dendriteImageSecR'
    spacing_file = '/home/lyf/data/seu_mouse/crop_data/scripts/AllbrainResolutionInfo.csv'
    data_list = load_data(data_dir, spacing_file)
    print(f'--> Number of data sample: {len(data_list)}')
    '''

    # load the spacing file
    spacing_dict = load_spacing(spacing_file)
    # get all annotated data
    data_list = []
    for brain_dir in glob.glob(os.path.join(data_dir, 'tiff', '*')):
        brain_id = os.path.split(brain_dir)[-1]
        print('--> Loading for brain: {:s}'.format(brain_id))
        spacing = spacing_dict[int(brain_id)]
        swc_dir = os.path.join(data_dir, 'swc', brain_id)
        for imgfile in glob.glob(os.path.join(brain_dir, '*.tiff')):
            prefix = get_file_prefix(imgfile)
            if is_train:
                swc_file = os.path.join(swc_dir, f'{prefix}.swc')
            else:
                swc_file = None
            data_list.append((imgfile, swc_file, spacing))

    return data_list

def calc_spacing_anisotropy(spacing):
    """
    spacing in (z,y,x) format
    """
    assert spacing[1] == spacing[2] and spacing[1] <= spacing[0], "Spacing in X- and Y-dimension must be the same, and the must smaller than Z-axis"
    spacing_multi = 1.0 * spacing[0] / spacing[2]
    return spacing_multi


class GenericPreprocessor(object):
    def __init__(self, separate_z_thresh=2):
        self.separate_z_thresh = separate_z_thresh

    def remove_nans(self, data):
        # inplace modification of nans
        data[np.isnan(data)] = 0
        return data

    def normalize(self, data, mask=None):
        assert data.ndim == 4, "image must in 4 dimension: c,z,y,x"
        for c in range(data.shape[0]):
            if mask is None:
                data[c] = (data[c] - data[c].mean()) / (data[c].std() + 1e-8)
            else:
                data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
                data[c][mask==0] = 0
        return data

    def resampling(self, data, tree=None, orig_spacing=None, target_spacing=None, order=3):
        """
        Unit test code: 
        imagefile = '/home/lyf/data/seu_mouse/crop_data/dendriteImageSecR/tiff/17302/8660_10980_2216.tiff'
        swcfile = '/home/lyf/data/seu_mouse/crop_data/dendriteImageSecR/swc/17302/8660_10980_2216.swc'
        image = sitk.GetArrayFromImage(sitk.ReadImage(imagefile))[None]
        tree = parse_swc(swcfile)
        orig_spacing = [0.2, 0.2, 1.0][::-1]
        target_spacing = [0.23, 0.23, 1.0][::-1]
        gp = GenericPreprocessor()
        image, tree = gp.resampling(image, tree, orig_spacing, target_spacing)
        print(f'image size: {image[0].shape}')
        
        sitk.WriteImage(sitk.GetImageFromArray(image[0]), '8660_10980_2216_resized.tiff')
        write_swc(tree, '8660_10980_2216_resized.swc')
        """

        assert data.ndim == 4, "image must in 4 dimension: c,z,y,x"
        #import ipdb; ipdb.set_trace()

        # whether resize separately for Z-axis
        separate_z = False
        if calc_spacing_anisotropy(orig_spacing) > self.separate_z_thresh or calc_spacing_anisotropy(target_spacing) > self.separate_z_thresh:
            separate_z = True
        
        dtype = data.dtype
        data = data.astype(np.float32)  # float32 is sufficient
        shape = np.array(data[0].shape)
        new_shape = np.round(((np.array(orig_spacing) / np.array(target_spacing)).astype(np.float32) * shape)).astype(int)
        #print(new_shape)
        # do resizing
        if np.all(shape == new_shape):
            print('no resampling necessary')
            return data, tree
        else:
            print('resampling...')

        if separate_z:
            z_axis = 0
            new_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[z_axis]):
                    reshaped_data.append(resize(data[c,slice_id], new_shape[1:], order=order, mode='edge'))
                reshaped_data = np.stack(reshaped_data, z_axis)
                if shape[z_axis] != new_shape[z_axis]:
                    # resizing in z dimension, code from sklearn's resize
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim 

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5 
                    map_cols = col_scale * (map_cols + 0.5) - 0.5 
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5 

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    new_data.append(map_coordinates(reshaped_data, coord_map, order=0, cval=0, mode='nearest')[None])
                else:
                    new_data.append(reshaped_data[None])
            new_data = np.vstack(new_data)
        else:
            print('no separate z, order', order)
            new_data = []
            for c in range(data.shape[0]):
                new_data.append(resize(data[c], new_shape, order, cval=0, mode='edge'))
            new_data = np.vstack(new_data)

        if tree is not None:
            # processing for the tree, scales in (z,y,x) order
            scales = np.array(orig_spacing) / np.array(target_spacing)
            cz0, cy0, cx0 = shape / 2.
            cz, cy, cx = new_shape / 2.
            new_tree = []
            for leaf in tree:
                idx, type_, x, y, z, rad, pid = leaf
                x = (x - cx0) * scales[2] + cx
                y = (y - cy0) * scales[1] + cy
                z = (z - cz0) * scales[0] + cz
                new_tree.append((idx, type_, x, y, z, rad, pid))
            
            return new_data.astype(dtype), new_tree
        else:
            return new_data.astype(dtype), tree

    def _preprocess_sample(self, imgfile, swcfile, imgfile_out, swcfile_out, spacing, target_spacing):
        print(f'--> Processing for image: {imgfile}')
        # load the image and annotated tree
        image = sitk.GetArrayFromImage(sitk.ReadImage(imgfile))
        if image.ndim == 3:
            image = image[None]
        tree = None
        if swcfile is not None:
            tree = parse_swc(swcfile)
        # remove nans
        image = self.remove_nans(image)
        # resampling to target spacing
        image, tree = self.resampling(image, tree, spacing, target_spacing)
        # normalize the image
        image = self.normalize(image, mask=None)
        # write the image and tree as well
        np.save(imgfile_out, image)
        if tree is not None:
            write_swc(tree, swcfile_out)

    def run(self, data_dir, spacing_file, output_dir, is_train=True, num_threads=8):
        print('Processing for dataset, should be run at least once for each dataset!')
        # get all files
        data_list = load_data(data_dir, spacing_file, is_train=is_train)
        print(f'Total number of samples found: {len(data_list)}')
        # estimate the target spacing
        spacings = [spacing for _,_,spacing in data_list]
        # assume spacing in format[z,y,x]
        spacings = sorted(spacings, key=lambda x: x.prod())
        target_spacing = spacings[len(spacings) // 2]


        maybe_mkdir_p(output_dir)
        # execute preprocessing
        args_list = []
        for imgfile, swcfile, spacing in data_list:
            prefix = get_file_prefix(imgfile)
            #print(imgfile, swcfile)
            imgfile_out = os.path.join(output_dir, f'{prefix}.npy')
            swcfile_out = os.path.join(output_dir, f'{prefix}.swc')
            args = imgfile, swcfile, imgfile_out, swcfile_out, spacing, target_spacing
            args_list.append(args)

        for args in args_list:
            self._preprocess_sample(*args)

        # execute in parallel
        #pt = Pool(num_threads)
        #pt.starmap(self._preprocess_sample, args_list)
        #pt.close()
        #pt.join()


    def dataset_split(self, task_dir, val_ratio=0.1, test_ratio=0.1, seed=1024, img_ext='npy', lab_ext='swc'):
        samples = []
        for imgfile in glob.glob(os.path.join(task_dir, '*.npy')):
            labfile = f'{imgfile[:len(img_ext)]}{lab_ext}'
            samples.append((imgfile, labfile))
        # data splitting
        num_tot = len(samples)
        num_val = int(round(num_tot * val_ratio))
        num_test = int(round(num_tot * test_ratio))
        num_train = num_tot - num_val - num_test
        print(f'Number of samples of total/train/val/test are {num_tot}/{num_train}/{num_val}/{num_test}')
        
        np.random.seed(seed)
        np.random.shuffle(samples)
        splits = {}
        splits['train_samples'] = samples[:num_train]
        splits['val_samples'] = samples[num_train:num_train+val]
        splits['test_samples'] = samples[-num_test:]
        # write to file
        with open('data_splits.pkl', 'wb') as fp:
            pickle.dump(splits, fp)
        

        
if __name__ == '__main__':
    data_dir = '/home/lyf/data/seu_mouse/crop_data/dendriteImageSecR'
    spacing_file = '/home/lyf/data/seu_mouse/crop_data/scripts/AllbrainResolutionInfo.csv'
    output_dir = '/home/lyf/Research/auto_trace/neuronet/data/task0001_17302'
    is_train = True
    num_threads = 8
    gp = GenericPreprocessor()
    gp.run(data_dir, spacing_file, output_dir, is_train=is_train, num_threads=num_threads)
    #gp.dataset_split(output_dir, val_ratio=0.1, test_ratio=0.1, seed=1024, img_ext='npy', lab_ext='swc')
    

