#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : generic_augmentation.py
#   Author       : Yufeng Liu
#   Date         : 2021-04-02
#   Description  : Some codes are borrowed from ssd.pytorch: https://github.com/amdegroot/ssd.pytorch
#
#================================================================

import numpy as np
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import SimpleITK as sitk
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, interpolate_img, rotate_coords_2d, rotate_coords_3d, scale_coords, elastic_deform_coordinates_2, resize_multichannel_image

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


# Coordinate-invariant augmentation
class RandomSaturation(AbstractTransform):
    def __init__(self, lower=0.8, upper=1.3, p=0.5):
        super(RandomSaturation, self).__init__(p)
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower"
        assert self.lower > 0, "contrast lower must be positive"

    def __call__(self, img, tree=None, spacing=None):
        if np.random.random() < self.p:
            img *= random.uniform(self.lower, self.upper)

        return img, tree, spacing

class RandomBrightness(AbstractTransform):
    def __init__(self, dratio=0.15, p=0.5):
        super(RandomBrightness, self).__init__(p)
        assert dratio >= 0. and dratio < 1.
        self.dratio = dratio

    def __call__(self, img, tree=None, spacing=None):
        if np.random.random() < self.p:
            img_flat = img.reshape((img.shape[0],-1))
            mm = img_flat.max() - img_flat.min()
            dmm = np.random.uniform(-self.dratio, self.dratio) * mm
            img += dmm.reshape((mm.shape[0],1,1,1))

        return img, tree, spacing

class RandomGaussianNoise(AbstractTransform):
    def __init__(self, max_var=0.2, p=0.5):
        super(RandomGaussianNoise, self).__init__(p)
        self.max_var = max_var

    def __call__(self, img, tree=None, spacing=None):
        if np.random.random() < self.p:
            var = np.random.uniform(0, self.max_var)
            img += np.random.normal(0, var, size=img.shape)
        return img, tree, spacing

class RandomGaussianBlur(AbstractTransform):
    def __init__(self, kernels=(0,1,2), p=0.5):
        super(RandomGaussianBlur, self).__init__(p)
        self.kernels = kernels
    
    def __call__(self, img, tree=None, spacing=None):
        assert spacing[0] >= spacing[1] and spacing[0] >= spacing[2]

        if np.random.random() < self.p:
            idx = np.random.randint(len(self.kernels))
            kernel = self.kernels[idx]
            kernel_z = max(int(round(kernel * (spacing[1] + spacing[2]) / spacing[0] + 1)), 1)
            kernel_xy = kernel * 2 + 1
            sigmas = (kernel_z, kernel_xy, kernel_xy)

            for c in range(img.shape[0]):
                img[c] = gaussian_filter(img[c], sigma=sigmas)
        return img, tree, spacing
            

class RandomResample(AbstractTransform):
    def __init__(self, p=0.5, zoom_range=(0.5,1), order_down=1, order_up=0, per_axis=True):
        super(RandomResample, self).__init__(p)
        self.zoom_range = zoom_range
        self.order_down = order_down
        self.order_up = order_up
        self.per_channel = per_channel
        
    def __call__(self, img, tree=None, spacing=None):
        if np.random.random() < self.p:
            if not img.dtype.name.startswith('float'):
                img = img.astype(np.float32)

            shape = np.array(img[0].shape)
            if per_axis:
                zoom = np.random.uniform(*self.zoom_range, size=len(shape))
            else:
                zoom = np.random.uniform(*self.zoom_range)
            target_shape = np.round(shape * zoom).astype(np.int)

            for c in range(img.shape[0]):
                downsampled = resize(img[c], target_shape, order=self.order_down, mode='edge', anti_aliasing=False)
                img[c] = resize(downsampled, shape, order=self.order_up, mode='edg', anti_aliasing=False)
    
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
        return img, tree, spacing
                

'''
The following geometric transformation can be composed into a unique geometric transformation
class RandomScale(object):

class RandomCrop(object)

class RandomPadding(object)

class RandomRotation(object)

class RandomShift(object)
'''

class RandomGeometric(AbstractTransform):
    def __init__(self, patch_size, p=1., patch_center_dist_from_border=30,
                 p_elastic_deform=0.2, deformation_scale=(0, 0.25),
                 p_rotation=0.2, angle_x=(0, 2*np.pi), 
                 angle_y=(0, 2*np.pi), angle_z=(0, 2*np.pi),
                 p_scale=0.2, scale=(0.75,1.25), axis_scale=False,
                 border_mode='nearest', border_cval=0,
                 order=3, random_crop=True):
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
        self.border_mode = border_mode
        self.border_cval = border_cval
        self.order = order
        self.random_crop = random_crop

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
            print(f'Elastic deformation with sigmas and mag: ', sigmas, mag)
            modified_coordinates = True

        if self.p_rotation > 0:
            if np.random.uniform() < self.p_rotation:
                a_x = np.random.uniform(self.angle_x[0], self.angle_x[1])
            else:
                a_x = 0
            if dim == 3:
                if np.random.uniform() < self.p_rotation:
                    a_y = np.random.uniform(self.angle_y[0], self.angle_y[1])
                else:
                    a_y = 0
                if np.random.uniform() < self.p_rotation:
                    a_z = np.random.uniform(self.angle_z[0], self.angle_z[1])
                else:
                    a_z = 0
                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            else:
                coords = rotate_coords_2d(coords, a_x)
            print(f'Rotation with with {a_x}, {a_y}, {a_y}')
            modified_coords = True

        if np.random.uniform() < self.p_scale:
            if self.axis_scale:
                sc = []
                for _ in range(dim):
                    if np.random.random() < 0.5 and self.scale[0] < 1:
                        sc.append(np.random.uniform(self.scale[0], 1))
                    else:
                        sc.append(np.random.uniform(max(self.scale[0],1), max(self.scale[1], 1)))
            else:
                if np.random.random() < 0.5 and self.scale[0] < 1:
                    sc = np.random.uniform(self.scale[0], 1)
                else:
                    sc = np.random.uniform(max(self.scale[0], 1), max(self.scale[1],1))
            coords = scale_coords(coords, sc)
            print(f'scaling with parameter: {sc}')
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
        return img, tree, spacing


if __name__ == '__main__':
    import time
    from neuronet.utils.image_util import normalize_normal, unnormalize_normal
    from neuronet.utils.util import set_deterministic

    
    file_prefix = '8315_19523_2299'
    imgfile = f'/home/lyf/data/seu_mouse/crop_data/dendriteImageSecR/tiff/17302/{file_prefix}.tiff'
    set_deterministic(True, seed=1024)
    img = sitk.GetArrayFromImage(sitk.ReadImage(imgfile))[None]
    img = img.astype(np.float32)
    # normalize to N(0,1) distribution
    img = normalize_normal(img)
    print(f'Statistics of original image: {img.mean()}, {img.std()}, {img.min()}, {img.max()}')
    tree = None
    spacing = None
    patch_size = [200, 480, 480]
    p_rotation = 1.0
    angle_x = (0, 2*np.pi)
    angle_y = (0, 0)
    angle_z = (0, 0)
    p_scale = 0.0
    random_crop = False
    p_elastic_deform = 0
    aug = RandomGeometric(patch_size, p=1., patch_center_dist_from_border=30,
                 p_elastic_deform=p_elastic_deform, deformation_scale=(0, 0.25),
                 p_rotation=p_rotation, angle_x=angle_x, 
                 angle_y=angle_y, angle_z=angle_z,
                 p_scale=p_scale, scale=(0.75,1.25), axis_scale=False,
                 border_mode='nearest', border_cval=0,
                 order=3, random_crop=random_crop)
    t0 = time.time()
    img_new, tree_new, spacing = aug(img, tree, spacing)
    print(f'Augmented image statistics: {img_new.mean()}, {img_new.std()}, {img_new.min()}, {img_new.max()}')
    print(f'Timed used: {time.time()-t0}s')
    # unnormalize for visual inspection
    img_unn = unnormalize_normal(img_new)
    print(f'Image statstics: {img_unn.mean()}, {img_unn.std()}, {img_unn.min()}, {img_unn.max()}')
    sitk.WriteImage(sitk.GetImageFromArray(img_unn[0]), f'{file_prefix}_aug.tiff')
