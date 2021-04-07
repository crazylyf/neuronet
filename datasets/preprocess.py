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
import pickle

from swc_handler import parse_swc, write_swc
from path_util import get_file_prefix

from neuronet.utils.image_util import normalize_normal
from neuronet.datasets.preprocess import soma_labelling, trim_swc, load_spacing

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
        image = normalize_normal(image, mask=None)
        # write the image and tree as well
        np.savez_compressed(imgfile_out, data=image.astype(np.float32))
        #np.save(imgfile_out, image.astype(np.float32))
        if tree is not None:
            write_swc(tree, swcfile_out)
            #with open(swcfile_out, 'wb') as fp:
            #    pickle.dump(tree, fp)

    @staticmethod
    def get_target_spacing(spacings):
        # assume spacing in format[z,y,x]
        spacings = sorted(spacings, key=lambda x: x.prod())
        target_spacing = spacings[len(spacings) // 2]
        return target_spacing

    def run(self, data_dir, spacing_file, output_dir, is_train=True, num_threads=8):
        print('Processing for dataset, should be run at least once for each dataset!')
        # get all files
        data_list = load_data(data_dir, spacing_file, is_train=is_train)
        print(f'Total number of samples found: {len(data_list)}')
        # estimate the target spacing
        spacings = [spacing for _,_,spacing in data_list]
        self.target_spacing = self.get_target_spacing(spacings)


        maybe_mkdir_p(output_dir)
        # execute preprocessing
        args_list = []
        for imgfile, swcfile, spacing in data_list:
            prefix = get_file_prefix(imgfile)
            #print(imgfile, swcfile)
            imgfile_out = os.path.join(output_dir, f'{prefix}.npz')
            #swcfile_out = os.path.join(output_dir, f'{prefix}.pkl')
            #imgfile_out = os.path.join(output_dir, f'{prefix}.npy')
            swcfile_out = os.path.join(output_dir, f'{prefix}.swc')
            args = imgfile, swcfile, imgfile_out, swcfile_out, spacing, self.target_spacing
            args_list.append(args)


        # execute in parallel
        pt = Pool(num_threads)
        pt.starmap(self._preprocess_sample, args_list)
        pt.close()
        pt.join()


    def dataset_split(self, task_dir, val_ratio=0.1, test_ratio=0.1, seed=1024, img_ext='npy', lab_ext='swc'):
        samples = []
        for imgfile in glob.glob(os.path.join(task_dir, f'*{img_ext}')):
            labfile = f'{imgfile[:-len(img_ext)]}{lab_ext}'
            samples.append((imgfile, labfile, self.target_spacing))
        # data splitting
        num_tot = len(samples)
        num_val = int(round(num_tot * val_ratio))
        num_test = int(round(num_tot * test_ratio))
        num_train = num_tot - num_val - num_test
        print(f'Number of samples of total/train/val/test are {num_tot}/{num_train}/{num_val}/{num_test}')
        
        np.random.seed(seed)
        np.random.shuffle(samples)
        splits = {}
        splits['train'] = samples[:num_train]
        splits['val'] = samples[num_train:num_train+num_val]
        splits['test'] = samples[-num_test:]
        # write to file
        with open(os.path.join(output_dir, 'data_splits.pkl'), 'wb') as fp:
            pickle.dump(splits, fp)
        

        
if __name__ == '__main__':
    data_dir = '/home/lyf/data/seu_mouse/crop_data/dendriteImageSecR'
    spacing_file = '/home/lyf/data/seu_mouse/crop_data/scripts/AllbrainResolutionInfo.csv'
    output_dir = '/home/lyf/Research/auto_trace/neuronet/data/task0001_17302'
    is_train = True
    num_threads = 8
    gp = GenericPreprocessor()
    gp.run(data_dir, spacing_file, output_dir, is_train=is_train, num_threads=num_threads)
    gp.dataset_split(output_dir, val_ratio=0.1, test_ratio=0.1, seed=1024, img_ext='npz', lab_ext='swc')
    

