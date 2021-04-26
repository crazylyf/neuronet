#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : generic_augmentation.py
#   Author       : Yufeng Liu
#   Date         : 2021-04-02
#   Description  : Some codes are borrowed from ssd.pytorch: https://github.com/amdegroot/ssd.pytorch
#                  And some augmentation implementations are directly copied from nnUNet.
#
#================================================================

import shutil
import numpy as np
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import SimpleITK as sitk
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, interpolate_img, rotate_coords_2d, rotate_coords_3d, scale_coords, elastic_deform_coordinates_2, resize_multichannel_image

from neuronet.utils import image_util

def get_random_shape(img, scale_range, per_axis):
    if type(img) == np.ndarray and img.size > 1024:
        shape = np.array(img[0].shape)
    else:
        shape = np.array(list(img))
    if per_axis:
        scales = np.random.uniform(*scale_range, size=len(shape))
    else:
        scales = np.array([np.random.uniform(*scale_range)] * len(shape))
    target_shape = np.round(shape * scales).astype(np.int)
    return shape, target_shape

def image_scale_4D(img, tree, spacing, shape, target_shape, mode, anti_aliasing, update_spacing):
    if target_shape.prod() / shape.prod() > 1:
        # up-scaling
        order = 0
    else:
        order = 1
    
    new_img = np.zeros((img.shape[0], *target_shape), dtype=img.dtype)
    for c in range(img.shape[0]):
        new_img[c] = resize(img[c], target_shape, order=order, mode=mode, anti_aliasing=anti_aliasing)
    
    # processing for the tree structure
    if tree is not None:
        scales = target_shape / shape
        new_tree = []
        for leaf in tree:
            idx, type_, x, y, z, r, p = leaf
            new_tree.append((idx,type_,x*scales[2],y*scales[1],z*scales[0],r,p))
        tree = new_tree

    # for the spacing
    if spacing is not None:
        if update_spacing:
            spacing = scales * np.array(spacing)
    return new_img, tree, spacing

def random_crop_image_4D(img, tree, spacing, target_shape):
    new_img = np.zeros((img.shape[0], *target_shape), dtype=img.dtype)
    for c in range(img.shape[0]):
        if c == 0:
            new_img[c],sz,sy,sx = image_util.random_crop_3D_image(img[c], target_shape)
        else:
            new_img[c] = img[sz:sz+target_shape[0], sy:sy+target_shape[2], sx:sx:target_shape[2]]
    # processing the tree
    if tree is not None:
        new_tree = []
        for leaf in tree:
            idx, type_, x, y, z, r, p = leaf
            x = x - sx
            # Since the y-coordinate in swc file is mirrored by crop
            # center, the coordinate change should take care!!!
            y = target_shape[1] - (img.shape[2] - y - sy)
            z = z - sz
            new_tree.append((idx,type_,x,y,z,r,p))
        return new_img, new_tree, spacing
    return new_img, tree, spacing


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, tree=None, spacing=None):
        for t in self.transforms:
            img, tree, spacing = t(img, tree, spacing)
        return img, tree, spacing


class AbstractTransform(object):
    def __init__(self, p=0.5):
        self.p = p
        
class ConvertToFloat(object):
    """
    Most augmentation assumes the input image of float type, so it is always recommended to 
    call this class before all augmentations.
    """
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
    
    def __call__(self, img, tree=None, spacing=None):
        if not img.dtype.name.startswith('float'):
            img = img.astype(self.dtype)
        return img, tree, spacing


# Coordinate-invariant augmentation
class RandomSaturation(AbstractTransform):
    def __init__(self, lower=0.9, upper=1.1, p=0.5):
        super(RandomSaturation, self).__init__(p)
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower"
        assert self.lower > 0, "contrast lower must be positive"

    def __call__(self, img, tree=None, spacing=None):
        if np.random.random() < self.p:
            sf = np.random.uniform(self.lower, self.upper)
            #print(f'RandomSaturation with factor: {sf}')
            img *= sf

        return img, tree, spacing

class RandomBrightness(AbstractTransform):
    def __init__(self, dratio=0.1, p=0.5):
        super(RandomBrightness, self).__init__(p)
        assert dratio >= 0. and dratio < 1.
        self.dratio = dratio

    def __call__(self, img, tree=None, spacing=None):
        if np.random.random() < self.p:
            img_flat = img.reshape((img.shape[0],-1))
            mm = img_flat.max(axis=1) - img_flat.min(axis=1)
            dmm = np.random.uniform(-self.dratio, self.dratio) * mm
            #print(f'RandomBrightness with shift: {dmm}')
            img += dmm.reshape((mm.shape[0],1,1,1))

        return img, tree, spacing

class RandomGaussianNoise(AbstractTransform):
    def __init__(self, max_var=0.2, p=0.5):
        super(RandomGaussianNoise, self).__init__(p)
        self.max_var = max_var

    def __call__(self, img, tree=None, spacing=None):
        if np.random.random() < self.p:
            var = np.random.uniform(0, self.max_var)
            img_flat = img.reshape((img.shape[0],-1))
            mm = img_flat.max(axis=1) - img_flat.min(axis=1)
            noise = np.random.normal(0, var, size=img.shape) * mm.reshape((-1,1,1,1))
            #print(f'RandomGaussianNoise with var: {var}')
            img += noise
        return img, tree, spacing

class RandomGaussianBlur(AbstractTransform):
    def __init__(self, kernels=(0,1), p=0.5):
        super(RandomGaussianBlur, self).__init__(p)
        self.kernels = kernels
    
    def __call__(self, img, tree=None, spacing=None):
        assert spacing[0] >= spacing[1] and spacing[0] >= spacing[2]

        if np.random.random() < self.p:
            idx = np.random.randint(len(self.kernels))
            kernel = self.kernels[idx]
            kernel_z = kernel * (spacing[1] + spacing[2]) / spacing[0] / 2
            kernel_z = max(int(round(kernel_z)) * 2 + 1, 1)
            kernel_xy = kernel * 2 + 1
            sigmas = (kernel_z, kernel_xy, kernel_xy)
            #print(f'RandomGaussianBlur with sigmas: {sigmas}')

            for c in range(img.shape[0]):
                img[c] = gaussian_filter(img[c], sigma=sigmas)
        return img, tree, spacing
            

class RandomResample(AbstractTransform):
    def __init__(self, p=0.5, zoom_range=(0.8,1), order_down=1, order_up=0, per_axis=True):
        super(RandomResample, self).__init__(p)
        self.zoom_range = zoom_range
        self.order_down = order_down
        self.order_up = order_up
        self.per_axis = per_axis
        
    def __call__(self, img, tree=None, spacing=None):
        if np.random.random() < self.p:
            shape, target_shape = get_random_shape(img, self.zoom_range, self.per_axis)

            #print(f'RandomSample with zoom factor: {zoom}')
            for c in range(img.shape[0]):
                downsampled = resize(img[c], target_shape, order=self.order_down, mode='edge', anti_aliasing=False)
                img[c] = resize(downsampled, shape, order=self.order_up, mode='edge', anti_aliasing=False)
    
        return img, tree, spacing
            
            
        
# Coordinate-changing augmentations
class RandomMirror(AbstractTransform):
    def __init__(self, p=0.5):
        super(RandomMirror, self).__init__(p)
        

    def __call__(self, img, tree=None, spacing=None):
        if np.random.random() < self.p:
            axis = np.random.randint(img.ndim - 1) + 1
            # NOTE: img in (c,z,y,x) order, while coord in tree is (x,y,z)
            if axis == 1:
                img = img[:,::-1,...]
            elif axis == 2:
                img = img[:,:,::-1,...]
            elif axis == 3:
                img = img[:,:,:,::-1]
            else:
                raise ValueError('Number of dimension should not exceed 4')
            if tree is not None:
                # processing tree structure
                shape = img[0].shape
                shape_axis = shape[axis-1]
                new_tree = []
                if axis == 1:
                    for leaf in tree:
                        idx, type_, x, y, z, r, p = leaf
                        z = shape_axis - z
                        new_tree.append((idx,type_,x,y,z,r,p))
                elif axis == 2:
                    for leaf in tree:
                        idx, type_, x, y, z, r, p = leaf
                        y = shape_axis - y
                        new_tree.append((idx,type_,x,y,z,r,p))
                else:
                    for leaf in tree:
                        idx, type_, x, y, z, r, p = leaf
                        x = shape_axis - x
                        new_tree.append((idx,type_,x,y,z,r,p))
                tree = new_tree
            #print(f'Mirroring for axis: {axis}')
        return img, tree, spacing
                

#The following geometric transformation can be composed into an unique geometric transformation. 
# But I prefer to use this separate versions, since they are implemented with matrix production, 
# which is much more efficient. 
class RandomScale(AbstractTransform):
    def __init__(self, p=0.5, scale_range=(0.85,1.25), per_axis=True, anti_aliasing=False, mode='edge', update_spacing=True):
        super(RandomScale, self).__init__(p)
        self.per_axis = per_axis
        self.anti_aliasing = anti_aliasing
        self.mode = mode
        self.update_spacing = update_spacing

    def __call__(self, img, tree=None, spacing=None):
        if np.random.random() < self.p:
            shape, target_shape = get_random_shape(img, self.scale_range, self.per_axis)
            img, tree, spacing = image_scale_4D(img, tree, spacing, shape, target_shape, self.mode, self.anti_aliasing, self.update_spacing)

        return img, tree, spacing

# verified
class ScaleToFixedSize(AbstractTransform):
    def __init__(self, p, target_shape, anti_aliasing=False, mode='edge', update_spacing=True):
        super(ScaleToFixedSize, self).__init__(p)
        self.target_shape = np.array(target_shape)
        self.anti_aliasing = anti_aliasing
        self.mode = mode
        self.update_spacing = update_spacing

    def __call__(self, img, tree=None, spacing=None):
        if np.random.random() < self.p:
            shape = np.array(img[0].shape)
            img, tree, spacing = image_scale_4D(img, tree, spacing, shape, self.target_shape, self.mode, self.anti_aliasing, self.update_spacing)

        return img, tree, spacing
            
    
class RandomCrop(AbstractTransform):
    def __init__(self, p=0.5, imgshape=None, crop_range=(0.85, 1), per_axis=True):
        super(RandomCrop, self).__init__(p)
        self.imgshape = imgshape
        self.crop_range = crop_range
        self.per_axis = per_axis

    def __call__(self, img, tree=None, spacing=None):
        if np.random.random() > self.p:
            return img, tree, spacing
        
        if self.crop_range[0] == self.crop_range[1]:
            target_shape = self.imgshape
        else:
            shape, target_shape = get_random_shape(self.imgshape, self.crop_range, self.per_axis)

        img, tree, spacing = random_crop_image_4D(img, tree, spacing, target_shape)
        return img, tree, spacing       

#verified
class RandomCenterCrop(AbstractTransform):
    def __init__(self, p=1.0, reference_shape=None, max_aniso_scale=1.2):
        super(RandomCenterCrop, self).__init__(p)
        self.reference_shape = reference_shape
        self.max_aniso_scale = max_aniso_scale

    def __call__(self, img, tree=None, spacing=None):
        if np.random.random() > self.p:
            return img, tree, spacing

        shape = np.array(img[0].shape)
        reference_shape = np.array(self.reference_shape)
        scales = shape / self.reference_shape
        min_dim = np.argmin(scales)

        target_shape = reference_shape.copy()
        for i in range(len(target_shape)):
            min_size = int(round(scales[min_dim] * reference_shape[i]))
            max_size = min(shape[i], int(self.max_aniso_scale * min_size))
            target_shape[i] = np.random.randint(min_size, max_size+1)
        target_shape = target_shape.astype(np.int)
        # do center cropping
        sz,sy,sx = (shape - target_shape) // 2
        img = img[:,sz:sz+target_shape[0],sy:sy+target_shape[1],sx:sx+target_shape[2]]
        
        if tree is not None:
            new_tree = []
            for leaf in tree:
                idx, type_, x, y, z, r, p = leaf
                x = x - sx
                y = y - sy
                z = z - sz
                new_tree.append((idx,type_,x,y,z,r,p))
            return img, new_tree, spacing
        else:
            return img, tree, spacing

# verified
class CenterCrop(AbstractTransform):
    def __init__(self, p=1.0, reference_shape=None):
        super(CenterCrop, self).__init__(p)
        self.reference_shape = reference_shape

    def __call__(self, img, tree=None, spacing=None):
        if np.random.random() > self.p:
            return img, tree, spacing

        shape = np.array(img[0].shape)
        reference_shape = np.array(self.reference_shape)
        scales = shape / self.reference_shape
        min_dim = np.argmin(scales)
        target_shape = np.round(scales[min_dim] * reference_shape).astype(int)
        # do center cropping
        sz,sy,sx = (shape - target_shape) // 2
        img = img[:,sz:sz+target_shape[0],sy:sy+target_shape[1],sx:sx+target_shape[2]]
        
        if tree is not None:
            new_tree = []
            for leaf in tree:
                idx, type_, x, y, z, r, p = leaf
                x = x - sx
                y = y - sy
                z = z - sz
                new_tree.append((idx,type_,x,y,z,r,p))
            return img, new_tree, spacing
        else:
            return img, tree, spacing
        

# NOTE: not verified, take care!
class RandomPadding(AbstractTransform):
    def __init__(self, p=0.5, pad_range=(1, 1.2), per_axis=True, pad_value=None):
        super(RandomPadding, self).__init__(p)
        self.pad_range = pad_range
        assert pad_range[0] >= 1 and pad_range[1] >= pad_range[0]
        self.per_axis = per_axis
        self.pad_value = pad_value

    def __call__(self, img, tree=None, spacing=None):
        if np.random.random() > self.p:
            return img, tree, spacing
        
        shape, target_shape = get_random_shape(img, self.pad_range, self.per_axis)
        for si, ti in zip(shape, target_shape):
            assert si <= ti
        if self.pad_value is None:
            # use lowerest value
            pad_value = img.min()
        else:
            pad_value = self.pad_value

        new_img = np.ones((img.shape[0], *target_shape), dtype=img.dtype) * pad_value
        #import ipdb; ipdb.set_trace()
        sz = np.random.randint(0, target_shape[0] - shape[0])
        sy = np.random.randint(0, target_shape[1] - shape[1])
        sx = np.random.randint(0, target_shape[2] - shape[2])
        for c in range(len(new_img)):
            new_img[c][sz:sz+shape[0], sy:sy+shape[1], sx:sx+shape[2]] = img
        if tree is not None:
            # for tree
            new_tree = []
            for leaf in tree:
                idx, type_, x, y, z, r, p = leaf
                x = x + sx
                y = y + sy
                z = z + sz
                new_tree.append((idx,type_,x,y,z,r,p))
            return new_img, new_tree, spacing
        else:
            return new_img, tree, spacing

class RandomRotation(AbstractTransform):
    def __init__(self, p=0.5):
        super(RandomRotation, self).__init__(p)
        
    def __call__(self, img, tree=None, spacing=None):
        raise NotImplementedError

# RandomShift is an subset of composition of Crop and Padding, thus we do not need to implement it.
class RandomShift(AbstractTransform):
    def __init__(self, p=0.5):
        super(RandomShift, self).__init__(p)

    def __call__(self, img, tree=None, spacing=None):
        raise NotImplementedError


# This implementation is very slow, as it use dense interpolation, instead of matrix production
class RandomGeometric(AbstractTransform):
    def __init__(self, patch_size, p=1., patch_center_dist_from_border=30,
                 p_elastic_deform=0.2, deformation_scale=(0, 0.25),
                 p_rotation=0.2, angle_x=(0, 2*np.pi), 
                 angle_y=(0, 2*np.pi), angle_z=(0, 2*np.pi),
                 p_scale=0.2, scale=(0.75,1.25), axis_scale=False, 
                 p_axis_scale=0.2,
                 border_mode='nearest', border_cval=0,
                 order=3, random_crop=True, anisotropic_thresh=2.):
        super(RandomGeometric, self).__init__(p)
        self.patch_size = patch_size    # in (z,y,x) order
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.p_elastic_deform = p_elastic_deform
        self.deformation_scale = deformation_scale
        self.p_rotation = p_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.p_scale = p_scale
        self.scale = scale
        self.axis_scale = axis_scale
        self.p_axis_scale = p_axis_scale
        self.border_mode = border_mode
        self.border_cval = border_cval
        self.order = order
        self.random_crop = random_crop
        self.anisotropic_thresh = anisotropic_thresh

    def __call__(self, img, tree=None, spacing=None):
        if np.random.random() > self.p:
            return img, tree, spacing

        patch_size = self.patch_size
        dim = len(patch_size)
        if dim == 2:
            img_p = np.zeros((img.shape[0], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            img_p = np.zeros((img.shape[0], patch_size[0], patch_size[1], patch_size[2]), dtype=np.float32)

        if not isinstance(self.patch_center_dist_from_border, (list, tuple, np.ndarray)):
            self.patch_center_dist_from_border = dim * [self.patch_center_dist_from_border]

        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = False

        if np.random.uniform() < self.p_elastic_deform:
            mag = []
            sigmas = []
            # scale is in percent of patch_size
            def_scale = np.random.uniform(self.deformation_scale[0], self.deformation_scale[1])
            for d in range(img.ndim - 1):
                # relative def_scale to scale in pixels
                sigmas.append(def_scale * patch_size[d])
                # define max magnitude and min_magnitude
                max_magnitude = sigmas[-1] * (1/2.)
                min_magnitude = sigmas[-1] * (1/8.)
                mag_real = np.random.uniform(min_magnitude, max_magnitude)
                mag.append(mag_real)

            coordinates = elastic_deform_coordinates_2(coords, sigmas, mag)
            #print(f'Elastic deformation with sigmas and mag: ', sigmas, mag)
            modified_coordinates = True

        # Rotation augmentation. If anisotropic axes, we only rotate along z-axis
        if self.p_rotation > 0:
            if np.random.uniform() < self.p_rotation:
                a_y = np.random.uniform(self.angle_y[0], self.angle_y[1])
            else:
                a_y = 0
            if dim == 3:
                if np.random.uniform() < self.p_rotation:
                    a_x = np.random.uniform(self.angle_x[0], self.angle_x[1])
                else:
                    a_x = 0
                if np.random.uniform() < self.p_rotation:
                    a_z = np.random.uniform(self.angle_z[0], self.angle_z[1])
                else:
                    a_z = 0

                if spacing[0] / spacing[2] > self.anisotropic_thresh:
                    #print('Anisotropic axes!')
                    a_y = 0
                    a_x = 0
                coords = rotate_coords_3d(coords, a_z, a_y, a_x)
            else:
                coords = rotate_coords_2d(coords, a_y)
            #print(f'Rotation with {a_z/np.pi*180.0}, {a_y/np.pi*180.}, {a_x/np.pi*180.}')
            modified_coords = True

        if np.random.uniform() < self.p_scale:
            # larger scale, smaller object in image
            if self.axis_scale and self.p_axis_scale: 
                # axis-independent scaling
                sc = []
                for _ in range(dim):
                    if np.random.random() < 0.5 and self.scale[0] < 1:
                        sc.append(np.random.uniform(self.scale[0], 1))
                    else:
                        sc.append(np.random.uniform(max(self.scale[0],1), self.scale[1]))
            else:
                if np.random.random() < 0.5 and self.scale[0] < 1:
                    sc = np.random.uniform(self.scale[0], 1)
                else:
                    sc = np.random.uniform(max(self.scale[0], 1), self.scale[1])
            coords = scale_coords(coords, sc)
            #print(f'scaling with parameter: {sc}')
            modified_coords = True

        if modified_coords:
            coords_mean = coords.mean(axis=tuple(range(1, len(coords.shape))), keepdims=True)
            coords -= coords_mean

            for d in range(dim):
                if self.random_crop:
                    ctr = np.random.uniform(self.patch_center_dist_from_border[d], img.shape[d+1] - self.patch_center_dist_from_border[d])
                else:
                    ctr = int(np.round(img.shape[d+1] / 2.))
                coords[d] += ctr
            for channel_id in range(img.shape[0]):
                img_p[channel_id] = interpolate_img(img[channel_id], coords, self.order, self.border_mode, cval=self.border_cval)

        else:
            if self.random_crop:
                margin = [self.patch_center_dist_from_border[d] - patch_size[d]//2 for d in range(dim)]
                d, s = random_crop_aug(img, s, patch_size, margin)
            else:
                d, s = center_crop_aug(img, patch_size, s)
            img_p = d
        return img_p, tree, spacing

class InstanceAugmentation(object):
    def __init__(self, p=0.2, imgshape=(256,512,512), phase='train'):
        if phase == 'train':
            self.augment = Compose([
                ConvertToFloat(),
                RandomCrop(1.0, imgshape),
                RandomSaturation(p=p),
                RandomBrightness(p=p),
                #RandomGaussianBlur(p=p/2.),
                RandomResample(p=p),
                RandomMirror(p=p),
                ScaleToFixedSize(1.0, imgshape),
            ])
        elif phase == 'val' or phase == 'test':
            self.augment = Compose([
                ConvertToFloat(),
                #RandomCenterCrop(1.0, imgshape),
                RandomCrop(1.0, imgshape, crop_range=(1,1))
            ])
        else:
            raise NotImplementedError

    def __call__(self, img, tree=None, spacing=None):
        return self.augment(img, tree, spacing)


if __name__ == '__main__':
    import time
    from neuronet.utils.image_util import normalize_normal, unnormalize_normal
    from neuronet.utils.util import set_deterministic
    from swc_handler import parse_swc, write_swc

    
    file_prefix = '8315_19523_2299'
    imgfile = f'../data/task0005_cropAll/{file_prefix}.npz'
    swcfile = f'../data/task0005_cropAll/{file_prefix}.swc'
    #set_deterministic(True, seed=1024)
    #img = sitk.GetArrayFromImage(sitk.ReadImage(imgfile))[None]
    #img = img.astype(np.float32)
    # normalize to N(0,1) distribution
    #img = normalize_normal(img)

    img = np.load(imgfile)['data']    # 4D
    if 1:
        # save original image for visual inspection
        img_orig_un = unnormalize_normal(img.copy()).astype(np.uint8)[0]
        sitk.WriteImage(sitk.GetImageFromArray(img_orig_un), 'original.tiff')
        shutil.copy(swcfile, 'original.swc')

    print(f'Statistics of original image: {img.mean()}, {img.std()}, {img.min()}, {img.max()}')
    tree = parse_swc(swcfile)
    spacing = [1.0, 0.23, 0.23]
    t0 = time.time()
    
    #aug = InstanceAugmentation(p=1.0)
    aug = InstanceAugmentation(0.2, (128,128,128))
    img_new, tree_new, spacing = aug(img, tree, spacing)
    print(f'Augmented image statistics: {img_new.mean()}, {img_new.std()}, {img_new.min()}, {img_new.max()}')
    print(f'Timed used: {time.time()-t0}s')
    # unnormalize for visual inspection
    img_unn = unnormalize_normal(img_new)
    img_unn = img_unn.astype(np.uint8)
    print(f'Image statstics: {img_unn.mean()}, {img_unn.std()}, {img_unn.min()}, {img_unn.max()}')
    sitk.WriteImage(sitk.GetImageFromArray(img_unn[0]), f'{file_prefix}_aug.tiff')
    write_swc(tree_new, f'{file_prefix}_aug.swc')
