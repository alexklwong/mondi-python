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
import numpy as np
import data_utils


def load_triplet_image(path, normalize=True, data_format='CHW'):
    '''
    Load in triplet frames from path

    Arg(s):
        path : str
            path to image triplet
        normalize : bool
            if set, normalize to [0, 1]
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : image at t - 1
        numpy[float32] : image at t
        numpy[float32] : image at t + 1
    '''

    images = data_utils.load_image(
        path,
        normalize=normalize,
        data_format=data_format)

    image1, image0, image2 = np.split(images, indices_or_sections=3, axis=-1)
    return image1, image0, image2

def horizontal_flip(images_arr):
    '''
    Perform horizontal flip on each sample

    Arg(s):
        images_arr : list[np.array[float32]]
            list of N x C x H x W tensors
    Returns:
        list[np.array[float32]] : list of transformed N x C x H x W image tensors
    '''

    for i, image in enumerate(images_arr):
        if len(image.shape) != 3:
            raise ValueError('Can only flip C x H x W images in dataloader.')

        flipped_image = np.flip(image, axis=-1)
        images_arr[i] = flipped_image

    return images_arr

def random_crop(inputs, shape, intrinsics=None, crop_type=['none']):
    '''
    Apply crop to inputs e.g. images, depth and if available adjust camera intrinsics

    Arg(s):
        inputs : list[numpy[float32]]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        intrinsics : list[numpy[float32]]
            list of 3 x 3 camera intrinsics matrix
        crop_type : str
            none, horizontal, vertical, anchored, bottom
    Return:
        list[numpy[float32]] : list of cropped inputs
        list[numpy[float32]] : if given, 3 x 3 adjusted camera intrinsics matrix
    '''

    n_height, n_width = shape
    _, o_height, o_width = inputs[0].shape

    # Get delta of crop and original height and width
    d_height = o_height - n_height
    d_width = o_width - n_width

    # By default, perform center crop
    y_start = d_height // 2
    x_start = d_width // 2

    if 'horizontal' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            widths = [
                anchor * d_width for anchor in crop_anchors
            ]
            x_start = int(widths[np.random.randint(low=0, high=len(widths))])

        # Randomly select a crop location
        else:
            x_start = np.random.randint(low=0, high=d_width+1)

    # If bottom alignment, then set starting height to bottom position
    if 'bottom' in crop_type:
        y_start = d_height

    elif 'vertical' in crop_type and np.random.rand() <= 0.30:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.50, 1.0
            ]

            heights = [
                anchor * d_height for anchor in crop_anchors
            ]
            y_start = int(heights[np.random.randint(low=0, high=len(heights))])

        # Randomly select a crop location
        else:
            y_start = np.random.randint(low=0, high=d_height+1)

    # Crop each input into (n_height, n_width)
    y_end = y_start + n_height
    x_end = x_start + n_width
    outputs = [
        T[:, y_start:y_end, x_start:x_end] for T in inputs
    ]

    # Adjust intrinsics
    if intrinsics is not None:
        offset_principal_point = [[0.0, 0.0, -x_start],
                                  [0.0, 0.0, -y_start],
                                  [0.0, 0.0, 0.0     ]]

        intrinsics = [
            in_ + offset_principal_point for in_ in intrinsics
        ]

        return outputs, intrinsics
    else:
        return outputs


class DepthCompletionInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) sparse depth
        (3) intrinsic camera calibration matrix

    Arg(s):
        image_paths : list[str]
            paths to images
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to intrinsic camera calibration matrix
        load_image_triplets : bool
            Whether or not inference images are stored as triplets or single
    '''

    def __init__(self,
                 image_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 load_image_triplets=False):

        self.n_sample = len(image_paths)

        for paths in [sparse_depth_paths, intrinsics_paths]:
            assert len(paths) == self.n_sample

        self.image_paths = image_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths

        self.data_format = 'CHW'
        self.load_image_triplets = load_image_triplets

    def __getitem__(self, index):

        # Load image
        if self.load_image_triplets:
            _, image, _ = load_triplet_image(
                path=self.image_paths[index],
                normalize=False,
                data_format=self.data_format)
        else:
            image = data_utils.load_image(
                path=self.image_paths[index],
                normalize=False,
                data_format=self.data_format)

        # Load sparse depth
        sparse_depth = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        intrinsics = np.load(self.intrinsics_paths[index])

        # Convert to float32
        image, sparse_depth, intrinsics = [
            T.astype(np.float32)
            for T in [image, sparse_depth, intrinsics]
        ]
        return image, sparse_depth, intrinsics

    def __len__(self):
        return self.n_sample


class MonitoredDistillationTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) camera image at t
        (2) left camera image at t - 1
        (3) left camera image at t + 1
        (4) if stereo is available, stereo camera image
        (5) sparse depth map at t
        (6) teacher output from ensemble at t
        (7) if stereo is available, intrinsic camera calibration matrix
        (8) if stereo is available, focal length and baseline

    Arg(s):
        image0_paths : list[str]
            paths to left camera images
        image1_paths : list[str]
            paths to right camera images
        sparse_depth0_paths : list[str]
            paths to left camera sparse depth maps
        sparse_depth1_paths : list[str]
            paths to right camera sparse depth maps
        ground_truth0_paths : list[str]
            paths to left camera ground truth depth maps
        ground_truth1_paths : list[str]
            paths to right camera ground truth depth maps
        ensemble_teacher_output0_paths : list[list[str]]
            list of lists of paths to left camera teacher output for ensemble
        ensemble_teacher_output1_paths : list[list[str]]
            list of lists of paths to right camera teacher output for ensemble
        intrinsics0_paths : list[str]
            paths to intrinsic left camera calibration matrix
        intrinsics1_paths : list[str]
            paths to intrinsic right camera calibration matrix
        focal_length_baseline0_paths : list[str]
            paths to focal length and baseline for left camera
        focal_length_baseline1_paths : list[str]
            paths to focal length and baseline for right camera
        random_crop_shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
        random_swap : bool
            Whether to perform random swapping as data augmentation
    '''

    def __init__(self,
                 image0_paths,
                 image1_paths,
                 sparse_depth0_paths,
                 sparse_depth1_paths,
                 ground_truth0_paths,
                 ground_truth1_paths,
                 ensemble_teacher_output0_paths,
                 ensemble_teacher_output1_paths,
                 intrinsics0_paths,
                 intrinsics1_paths,
                 focal_length_baseline0_paths,
                 focal_length_baseline1_paths,
                 random_crop_shape=None,
                 random_crop_type=None,
                 random_swap=False):

        self.n_sample = len(image0_paths)

        # Make sure that all paths in stereo stream is present
        self.stereo_available = \
            image1_paths is not None and \
            sparse_depth1_paths is not None and \
            ground_truth1_paths is not None and \
            intrinsics1_paths is not None and \
            focal_length_baseline0_paths is not None and \
            focal_length_baseline1_paths is not None and \
            ensemble_teacher_output1_paths is not None and \
            None not in image1_paths and \
            None not in sparse_depth1_paths and \
            None not in ground_truth1_paths and \
            None not in intrinsics1_paths and \
            None not in focal_length_baseline0_paths and \
            None not in focal_length_baseline1_paths and \
            None not in ensemble_teacher_output1_paths

        # If it is missing then populate them with None
        if not self.stereo_available:
            image1_paths = [None] * self.n_sample
            sparse_depth1_paths = [None] * self.n_sample
            ground_truth1_paths = [None] * self.n_sample
            intrinsics1_paths = [None] * self.n_sample
            focal_length_baseline0_paths = [None] * self.n_sample
            focal_length_baseline1_paths = [None] * self.n_sample
            ensemble_teacher_output1_paths = \
                [[None] * self.n_sample] * len(ensemble_teacher_output0_paths)

        input_paths = [
            image1_paths,
            sparse_depth0_paths,
            sparse_depth1_paths,
            ground_truth0_paths,
            ground_truth1_paths,
            intrinsics0_paths,
            intrinsics1_paths,
            focal_length_baseline0_paths,
            focal_length_baseline1_paths
        ]

        input_paths = input_paths + \
            ensemble_teacher_output0_paths + \
            ensemble_teacher_output1_paths

        for paths in input_paths:
            assert len(paths) == self.n_sample

        self.image0_paths = image0_paths
        self.image1_paths = image1_paths

        self.sparse_depth0_paths = sparse_depth0_paths
        self.sparse_depth1_paths = sparse_depth1_paths

        self.ground_truth0_paths = ground_truth0_paths
        self.ground_truth1_paths = ground_truth1_paths

        self.intrinsics0_paths = intrinsics0_paths
        self.intrinsics1_paths = intrinsics1_paths

        self.focal_length_baseline0_paths = focal_length_baseline0_paths
        self.focal_length_baseline1_paths = focal_length_baseline1_paths

        self.ensemble_teacher_output0_paths = ensemble_teacher_output0_paths
        self.ensemble_teacher_output1_paths = ensemble_teacher_output1_paths

        self.random_crop_type = random_crop_type
        self.random_crop_shape = random_crop_shape

        self.do_random_crop = \
            random_crop_shape is not None and all([x > 0 for x in random_crop_shape])

        self.do_random_swap = random_swap and self.stereo_available

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Swap and flip a stereo video stream
        do_swap = True if self.do_random_swap and np.random.uniform() < 0.5 else False

        if do_swap:
            # Swap paths for 0 and 1 indices
            image0_path = self.image1_paths[index]
            sparse_depth0_path = self.sparse_depth1_paths[index]
            ground_truth0_path = self.ground_truth1_paths[index]
            ensemble_teacher_output0_paths = self.ensemble_teacher_output1_paths
            intrinsics0_path = self.intrinsics1_paths[index]
            focal_length_baseline0_path = self.focal_length_baseline1_paths[index]

            image3_path = self.image0_paths[index]
        else:
            # Keep paths consistent
            image0_path = self.image0_paths[index]
            sparse_depth0_path = self.sparse_depth0_paths[index]
            ground_truth0_path = self.ground_truth0_paths[index]
            ensemble_teacher_output0_paths = self.ensemble_teacher_output0_paths
            intrinsics0_path = self.intrinsics0_paths[index]
            focal_length_baseline0_path = self.focal_length_baseline0_paths[index]

            image3_path = self.image1_paths[index]

        # Load images at times: t-1, t, t+1
        image1, image0, image2 = load_triplet_image(
            path=image0_path,
            normalize=False,
            data_format=self.data_format)

        # Load sparse depth map at time t
        sparse_depth0 = data_utils.load_depth(
            path=sparse_depth0_path,
            data_format=self.data_format)

        # Load ground_truth map at time t
        ground_truth0 = data_utils.load_depth(
            path=ground_truth0_path,
            data_format=self.data_format)

        # Load teacher output from ensemble
        teacher_output0 = []

        for paths in ensemble_teacher_output0_paths:
            teacher_output0.append(
                data_utils.load_depth(
                    path=paths[index],
                    data_format=self.data_format))

        teacher_output0 = np.concatenate(teacher_output0, axis=0)

        # Load camera intrinsics
        intrinsics0 = np.load(intrinsics0_path)

        # Load stereo pair for image0
        if self.stereo_available:
            _, image3, _ = load_triplet_image(
                path=image3_path,
                normalize=False,
                data_format=self.data_format)

            # Load camera intrinsics
            focal_length_baseline0 = np.load(focal_length_baseline0_path)
        else:
            image3 = image0.copy()
            focal_length_baseline0 = np.array([0, 0])

        inputs = [
            image0,
            image1,
            image2,
            image3,
            sparse_depth0,
            ground_truth0,
            teacher_output0,
        ]

        # If we swapped L and R, also need to horizontally flip images
        if do_swap:
            inputs = horizontal_flip(inputs)

        # Crop input images and depth maps and adjust intrinsics
        if self.do_random_crop:
            inputs, [intrinsics0] = random_crop(
                inputs=inputs,
                shape=self.random_crop_shape,
                intrinsics=[intrinsics0],
                crop_type=self.random_crop_type)

        # Convert inputs to float32
        inputs = inputs + [intrinsics0, focal_length_baseline0]

        inputs = [
            T.astype(np.float32)
            for T in inputs
        ]

        return inputs

    def __len__(self):
        return self.n_sample
