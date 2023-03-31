'''
Authors:
Tian Yu Liu <tianyu@cs.ucla.edu>
Parth Agrawal <parthagrawal24@ucla.edu>
Allison Chen <allisonchen2@ucla.edu>
Alex Wong <alex.wong@yale.edu>

If you use this code, please cite the following paper:
T.Y. Liu, P. Agrawal, A. Chen, B.W. Hong, and A. Wong. Monitored Distillation for Positive Congruent Depth Completion.
https://arxiv.org/abs/2203.16034

@inproceedings{liu2022monitored,
  title={Monitored distillation for positive congruent depth completion},
  author={Liu, Tian Yu and Agrawal, Parth and Chen, Allison and Hong, Byung-Woo and Wong, Alex},
  booktitle={European Conference on Computer Vision},
  year={2022},
  organization={Springer}
}
'''

import torch
import torchvision.transforms.functional as functional
from PIL import Image
import numpy as np


class Transforms(object):

    def __init__(self,
                 normalized_image_range=[0, 255],
                 random_crop_to_shape=[-1, -1],
                 random_flip_type=['none'],
                 random_remove_points=[0.70, 0.70],
                 random_brightness=[-1],
                 random_contrast=[-1],
                 random_saturation=[-1]):
        '''
        Transforms and augmentation class

        Arg(s):
            normalized_image_range : list[float]
                intensity range after normalizing images
            random_crop_to_shape : list[int]
                if given [h, w] output shape after random crop, if [h1, w1, h2, w2] then min and max shape size
            random_flip_type : list[str]
                none, horizontal, vertical
            random_remove_points : list[float]
                percentage of points to remove in range map
            random_brightness : list[float]
                brightness adjustment [0, B], from 0 (black image) to B factor increase
            random_contrast : list[float]
                contrast adjustment [0, C], from 0 (gray image) to C factor increase
            random_saturation : list[float]
                saturation adjustment [0, S], from 0 (black image) to S factor increase
        '''

        # Image normalization
        self.normalized_image_range = normalized_image_range

        # Geometric augmentations
        self.do_random_horizontal_flip = True if 'horizontal' in random_flip_type else False
        self.do_random_vertical_flip = True if 'vertical' in random_flip_type else False

        self.do_random_remove_points = True if -1 not in random_remove_points else False
        self.remove_points_range = random_remove_points

        self.do_random_crop_to_shape = True if -1 not in random_crop_to_shape else False

        self.do_random_crop_to_shape_exact = False
        self.do_random_crop_to_shape_range = False

        if self.do_random_crop_to_shape:
            if len(random_crop_to_shape) == 2:
                # If performed, will only crop to one shape
                self.do_random_crop_to_shape_exact = True
                self.random_crop_to_shape_height = random_crop_to_shape[0]
                self.random_crop_to_shape_width = random_crop_to_shape[1]
            elif len(random_crop_to_shape) == 4:
                # If performed will crop to any shape between min and max shapes
                self.do_random_crop_to_shape_range = True
                self.random_crop_to_shape_height_min = random_crop_to_shape[0]
                self.random_crop_to_shape_width_min = random_crop_to_shape[1]
                self.random_crop_to_shape_height_max = random_crop_to_shape[2]
                self.random_crop_to_shape_width_max = random_crop_to_shape[3]
            else:
                raise ValueError('Unsupported input for random crop to shape: {}'.format(
                    random_crop_to_shape))

        # RGB Augmentations
        self.do_random_brightness = True if -1 not in random_brightness else False
        self.random_brightness = random_brightness
        self.do_random_contrast = True if -1 not in random_contrast else False
        self.random_contrast = random_contrast
        self.do_random_saturation = True if -1 not in random_saturation else False
        self.random_saturation = random_saturation

    def transform(self,
                  images_arr,
                  range_maps_arr=[],
                  validity_maps_arr=[],
                  intrinsics_arr=[],
                  random_transform_probability=0.50):
        '''
        Applies transform to images and ground truth

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            range_maps_arr : list[torch.Tensor]
                list of N x c x H x W tensors
            validity_maps_arr : list[torch.Tensor]
                list of N x c x H x W tensors
            intrinsics_arr : list[torch.Tensor]
                list of N x 3 x 3 tensors
            random_transform_probability : float
                probability to perform transform
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
            list[torch.Tensor[float32]] : list of transformed N x c x H x W range maps tensors
        '''

        device = images_arr[0].device

        n_dim = images_arr[0].ndim

        if n_dim == 4:
            n_batch, _, n_height, n_width = images_arr[0].shape
        elif n_dim == 5:
            n_batch, _, _, n_height, n_width = images_arr[0].shape
        else:
            raise ValueError('Unsupported number of dimensions: {}'.format(n_dim))

        do_random_transform = \
            np.random.rand(n_batch) <= random_transform_probability

        '''
        Photometric Transformations (only on images)
        '''
        for idx, images in enumerate(images_arr):
            # In case user pass in [0, 255] range image as float type
            if torch.max(images) > 1.0:
                images_arr[idx] = images.to(torch.uint8)

        if self.do_random_brightness:

            do_brightness = np.logical_and(
                do_random_transform,
                np.random.rand(n_batch) <= 0.50)

            values = torch.rand(n_batch, device=device)

            brightness_min, brightness_max = self.random_brightness
            factors = (brightness_max - brightness_min) * values + brightness_min

            images_arr = self.adjust_brightness(images_arr, do_brightness, factors)

        if self.do_random_contrast:

            do_contrast = np.logical_and(
                do_random_transform,
                np.random.rand(n_batch) <= 0.50)

            values = torch.rand(n_batch, device=device)

            contrast_min, contrast_max = self.random_contrast
            factors = (contrast_max - contrast_min) * values + contrast_min

            images_arr = self.adjust_contrast(images_arr, do_contrast, factors)

        if self.do_random_saturation:

            do_saturation = np.logical_and(
                do_random_transform,
                np.random.rand(n_batch) <= 0.50)

            values = torch.rand(n_batch, device=device)

            saturation_min, saturation_max = self.random_saturation
            factors = (saturation_max - saturation_min) * values + saturation_min

            images_arr = self.adjust_saturation(images_arr, do_saturation, factors)

        # Normalize images to a given range
        images_arr = self.normalize_images(
            images_arr,
            normalized_image_range=self.normalized_image_range)

        '''
        Geometric Transformations
        '''
        do_random_crop_to_shape = \
            self.do_random_crop_to_shape_exact and np.random.rand(1) <= 0.50 or \
            self.do_random_crop_to_shape_range

        if do_random_crop_to_shape:

            if self.do_random_crop_to_shape_exact:
                random_crop_to_shape_height = self.random_crop_to_shape_height
                random_crop_to_shape_width = self.random_crop_to_shape_width

            if self.do_random_crop_to_shape_range:
                random_crop_to_shape_height = np.random.randint(
                    low=self.random_crop_to_shape_height_min,
                    high=self.random_crop_to_shape_height_max + 1)

                random_crop_to_shape_width = np.random.randint(
                    low=self.random_crop_to_shape_width_min,
                    high=self.random_crop_to_shape_width_max + 1)

            # Random crop factors
            start_y = torch.randint(
                low=0,
                high=n_height - random_crop_to_shape_height + 1,
                size=(n_batch,),
                device=device)

            start_x = torch.randint(
                low=0,
                high=n_width - random_crop_to_shape_width + 1,
                size=(n_batch,),
                device=device)

            end_y = start_y + random_crop_to_shape_height
            end_x = start_x + random_crop_to_shape_width

            start_yx = [start_y, start_x]
            end_yx = [end_y, end_x]

            images_arr = self.crop(
                images_arr,
                start_yx=start_yx,
                end_yx=end_yx)

            range_maps_arr = self.crop(
                range_maps_arr,
                start_yx=start_yx,
                end_yx=end_yx)

            validity_maps_arr = self.crop(
                validity_maps_arr,
                start_yx=start_yx,
                end_yx=end_yx)

            intrinsics_arr = self.adjust_intrinsics(
                intrinsics_arr,
                scales=1,
                x_offsets=n_width - random_crop_to_shape_width,
                y_offsets=n_height - random_crop_to_shape_height)

            # Update shape of tensors after crop
            n_height = random_crop_to_shape_height
            n_width = random_crop_to_shape_width

        if self.do_random_horizontal_flip:

            do_horizontal_flip = np.logical_and(
                do_random_transform,
                np.random.rand(n_batch) <= 0.50)

            images_arr = self.horizontal_flip(
                images_arr,
                do_horizontal_flip)

            range_maps_arr = self.horizontal_flip(
                range_maps_arr,
                do_horizontal_flip)

            validity_maps_arr = self.horizontal_flip(
                validity_maps_arr,
                do_horizontal_flip)

        if self.do_random_vertical_flip:

            do_vertical_flip = np.logical_and(
                do_random_transform,
                np.random.rand(n_batch) <= 0.50)

            images_arr = self.vertical_flip(
                images_arr,
                do_vertical_flip)

            range_maps_arr = self.vertical_flip(
                range_maps_arr,
                do_vertical_flip)

            validity_maps_arr = self.vertical_flip(
                validity_maps_arr,
                do_vertical_flip)

        if self.do_random_remove_points:

            do_remove_points = np.logical_and(
                do_random_transform,
                np.random.rand(n_batch) <= 0.50)

            values = torch.rand(n_batch, device=device)

            remove_points_min, remove_points_max = self.remove_points_range

            densities = \
                (remove_points_max - remove_points_min) * values + remove_points_min

            range_maps_arr = self.remove_random_nonzero(
                images_arr=range_maps_arr,
                do_remove=do_remove_points,
                densities=densities)

        # Return the transformed inputs

        for idx, images in enumerate(images_arr):
            if torch.max(images) > 1.0:
                images_arr[idx] = images.float()

        outputs = []

        if len(images_arr) > 0:
            outputs.append(images_arr)

        if len(range_maps_arr) > 0:
            outputs.append(range_maps_arr)

        if len(validity_maps_arr) > 0:
            outputs.append(validity_maps_arr)

        if len(intrinsics_arr) > 0:
            outputs.append(intrinsics_arr)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    '''
    Photometric transforms
    '''
    def normalize_images(self, images_arr, normalized_image_range=[0, 1]):
        '''
        Normalize image to a given range

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            normalized_image_range : list[float]
                intensity range after normalizing images
        Returns:
            images_arr[torch.Tensor[float32]] : list of normalized N x C x H x W tensors
        '''

        if normalized_image_range == [0, 1]:
            images_arr = [
                images / 255.0 for images in images_arr
            ]
        elif normalized_image_range == [-1, 1]:
            images_arr = [
                2.0 * (images / 255.0) - 1.0 for images in images_arr
            ]
        elif normalized_image_range == [0, 255]:
            pass
        else:
            raise ValueError('Unsupported normalization range: {}'.format(
                normalized_image_range))

        return images_arr

    def adjust_brightness(self, images_arr, do_brightness, factors):
        '''
        Adjust brightness on each sample

        Args:
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_brightness : bool
                N booleans to determine if brightness is adjusted on each sample
            factors : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_brightness[b]:
                    images[b, ...] = functional.adjust_brightness(image, factors[b])

            images_arr[i] = images

        return images_arr

    def adjust_contrast(self, images_arr, do_contrast, factors):
        '''
        Adjust contrast on each sample

        Args:
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_contrast : bool
                N booleans to determine if contrast is adjusted on each sample
            factors : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_contrast[b]:
                    images[b, ...] = functional.adjust_contrast(image, factors[b])

            images_arr[i] = images

        return images_arr

    def adjust_saturation(self, images_arr, do_saturation, factors):
        '''
        Adjust saturation on each sample

        Args:
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_saturation : bool
                N booleans to determine if saturation is adjusted on each sample
            gammas : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_saturation[b]:
                    images[b, ...] = functional.adjust_saturation(image, factors[b])

            images_arr[i] = images

        return images_arr
    '''
    Geometric transforms
    '''
    def crop(self, images_arr, start_yx, end_yx):
        '''
        Performs cropping on images

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            start_yx : list[int, int]
                top left corner y, x coordinate
            end_yx : list
                bottom right corner y, x coordinate
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            images_cropped = []

            for b, image in enumerate(images):

                start_y = start_yx[0][b]
                start_x = start_yx[1][b]
                end_y = end_yx[0][b]
                end_x = end_yx[1][b]

                # Crop image
                image = image[..., start_y:end_y, start_x:end_x]

                images_cropped.append(image)

            images_arr[i] = torch.stack(images_cropped, dim=0)

        return images_arr

    def horizontal_flip(self, images_arr, do_horizontal_flip):
        '''
        Perform horizontal flip on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_horizontal_flip : bool
                N booleans to determine if horizontal flip is performed on each sample
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_horizontal_flip[b]:
                    images[b, ...] = torch.flip(image, dims=[-1])

            images_arr[i] = images

        return images_arr

    def vertical_flip(self, images_arr, do_vertical_flip):
        '''
        Perform vertical flip on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_vertical_flip : bool
                N booleans to determine if vertical flip is performed on each sample
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_vertical_flip[b]:
                    images[b, ...] = torch.flip(image, dims=[-2])

            images_arr[i] = images

        return images_arr

    def remove_random_nonzero(self, images_arr, do_remove, densities):
        '''
        Remove random nonzero for each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_remove : bool
                N booleans to determine if random remove is performed on each sample
            densities : float
                N floats to determine how much to remove from each sample
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_remove[b]:

                    nonzero_indices = self.random_nonzero(image, density=densities[b])
                    image[nonzero_indices] = 0.0

                    images[b, ...] = image

            images_arr[i] = images

        return images_arr

    def random_nonzero(self, T, density=0.10):
        '''
        Randomly selects nonzero elements

        Arg(s):
            T : torch.Tensor[float32]
                N x C x H x W tensor
            density : float
                percentage of nonzero elements to select
        Returns:
            list[tuple[torch.Tensor[float32]]] : list of tuples of indices
        '''

        # Find all nonzero indices
        nonzero_indices = (T > 0).nonzero(as_tuple=True)

        # Randomly choose a subset of the indices
        random_subset = torch.randperm(nonzero_indices[0].shape[0], device=T.device)
        random_subset = random_subset[0:int(density * random_subset.shape[0])]

        random_nonzero_indices = [
            indices[random_subset] for indices in nonzero_indices
        ]

        return random_nonzero_indices


    def adjust_intrinsics(self,
                          intrinsics_arr,
                          scales=[1.0],
                          x_offsets=[0],
                          y_offsets=[0]):
        '''
        Adjust the each camera intrinsics based on the provided scaling factors and offsets

        Arg(s):
            intrinsics : torch.Tensor[float32]
                3 x 3 camera intrinsics
            scales : list[float]
                scaling factor for focal lengths and optical centers
            x_offsets : list[int]
                amount of horizontal offset to SUBTRACT from optical center
            y_offsets : list[int]
                amount of vertical offset to SUBTRACT from optical center
        Returns:
            torch.Tensor[float32] : 3 x 3 adjusted camera intrinsics
        '''

        for i, intrinsics in enumerate(intrinsics_arr):

            length = len(intrinsics)
            scales = [scales] * length if not isinstance(scales, list) else scales
            x_offsets = [x_offsets] * length if not isinstance(x_offsets, list) else x_offsets
            y_offsets = [y_offsets] * length if not isinstance(y_offsets, list) else y_offsets

            for b, K in enumerate(intrinsics):

                scale = scales[b]
                x_offset = x_offsets[b]
                y_offset = y_offsets[b]

                # Scale and subtract offset
                K[0, 0] = K[0, 0] * scale
                K[0, 2] = K[0, 2] * scale - x_offset
                K[1, 1] = K[1, 1] * scale
                K[1, 2] = K[1, 2] * scale - y_offset

            intrinsics[b] = K

        return intrinsics_arr
