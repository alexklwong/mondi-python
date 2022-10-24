import numpy as np
import torch.utils.data
import data_utils


def load_image_triplet(path, normalize=True, data_format='CHW'):
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


class ScaffNetTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) sparse depth
        (2) target dense (ground truth) depth

    Args:
        sparse_depth_paths : list[str]
            paths to sparse depth
        dense_depth_paths : list[str]
            paths to dense depth
        cap_dataset_depth_method : str
            remove, set_to_max
        min_dataset_depth : int
            minimum depth to load, any values less will be set to 0.0
        max_dataset_depth : float
            maximum depth to load, any values more will be set to 0.0
        random_crop_shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
    '''

    def __init__(self,
                 sparse_depth_paths,
                 dense_depth_paths,
                 cap_dataset_depth_method='set_to_max',
                 min_dataset_depth=-1.0,
                 max_dataset_depth=-1.0,
                 random_crop_shape=None,
                 random_crop_type=None):

        self.sparse_depth_paths = sparse_depth_paths
        self.dense_depth_paths = dense_depth_paths

        self.n_sample = len(self.sparse_depth_paths)
        assert self.n_sample == len(self.dense_depth_paths)

        self.cap_dataset_depth_method = cap_dataset_depth_method
        self.min_dataset_depth = min_dataset_depth
        self.max_dataset_depth = max_dataset_depth

        self.random_crop_type = random_crop_type
        self.random_crop_shape = random_crop_shape

        self.do_random_crop = \
            random_crop_shape is not None and all([x > 0 for x in random_crop_shape])

        self.data_format = 'CHW'

    def __getitem__(self, index):
        # Load depth
        sparse_depth = data_utils.load_depth(
            self.sparse_depth_paths[index],
            data_format=self.data_format)

        dense_depth = data_utils.load_depth(
            self.dense_depth_paths[index],
            data_format=self.data_format)

        if self.do_random_crop:
            [sparse_depth, dense_depth] = random_crop(
                inputs=[sparse_depth, dense_depth],
                shape=self.random_crop_shape,
                crop_type=self.random_crop_type)

        if self.min_dataset_depth > 0.0:
            sparse_depth[sparse_depth < self.min_dataset_depth] = 0.0
            dense_depth[dense_depth < self.min_dataset_depth] = 0.0

        if self.max_dataset_depth > 0.0:
            if self.cap_dataset_depth_method == 'remove':
                sparse_depth[sparse_depth > self.max_dataset_depth] = 0.0
                dense_depth[dense_depth > self.max_dataset_depth] = 0.0
            elif self.cap_dataset_depth_method == 'set_to_max':
                sparse_depth[sparse_depth > self.max_dataset_depth] = self.max_dataset_depth
                dense_depth[dense_depth > self.max_dataset_depth] = self.max_dataset_depth

        sparse_depth, dense_depth = [
            T.astype(np.float32)
            for T in [sparse_depth, dense_depth]
        ]

        return sparse_depth, dense_depth

    def __len__(self):
        return self.n_sample

class ScaffNetInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching depth

    Arg(s):
        depth_paths : list[str]
            paths to depth maps
    '''

    def __init__(self, depth_paths):

        self.depth_paths = depth_paths

        self.data_format = 'CHW'

    def __getitem__(self, index):
        # Load depth
        depth = data_utils.load_depth(
            self.depth_paths[index],
            data_format=self.data_format)

        return depth.astype(np.float32)

    def __len__(self):
        return len(self.depth_paths)


class FusionNetTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image at time t-1, t, and t+1
        (2) dense depth at time t
        (3) sparse depth at time t
        (4) camera intrinsics matrix

    Arg(s):
        images_paths : list[str]
            paths to image triplets
        dense_depth_paths : list[str]
            paths to dense depth maps
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to 3 x 3 camera intrinsics matrix
        random_crop_shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
    '''

    def __init__(self,
                 images_paths,
                 dense_depth_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 random_crop_shape=None,
                 random_crop_type=['none']):

        self.images_paths = images_paths
        self.dense_depth_paths = dense_depth_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths

        self.n_sample = len(images_paths)

        for paths in [sparse_depth_paths, dense_depth_paths, intrinsics_paths]:
            assert len(paths) == self.n_sample

        self.random_crop_shape = random_crop_shape
        self.do_random_crop = \
            self.random_crop_shape is not None and all([x > 0 for x in self.random_crop_shape])

        # Augmentation
        self.random_crop_type = random_crop_type

        self.data_format = 'CHW'

    def __getitem__(self, index):
        # Load image at t-1, t, t+1
        image1, image0, image2 = load_image_triplet(
            self.images_paths[index],
            normalize=False,
            data_format=self.data_format)

        # Load sparse depth at time t
        sparse_depth0 = data_utils.load_depth(
            self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load dense depth at time t
        dense_depth0 = data_utils.load_depth(
            self.dense_depth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        intrinsics = np.load(self.intrinsics_paths[index]).astype(np.float32)

        inputs = [
            image0, image1, image2, dense_depth0, sparse_depth0
        ]

        # Crop image, depth and adjust intrinsics
        if self.do_random_crop:
            inputs, [intrinsics] = random_crop(
                inputs=inputs,
                shape=self.random_crop_shape,
                intrinsics=[intrinsics],
                crop_type=self.random_crop_type)

        # Convert to float32
        inputs = inputs + [intrinsics]

        inputs = [
            T.astype(np.float32)
            for T in inputs
        ]

        return inputs

    def __len__(self):
        return self.n_sample


class FusionNetInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) sparse depth

    Arg(s):
        image_paths : list[str]
            paths to image triplets
        sparse_depth_paths : list[str]
            paths to sparse depth maps
    '''

    def __init__(self,
                 image_paths,
                 dense_depth_paths,
                 sparse_depth_paths,
                 use_image_triplet=True):

        self.image_paths = image_paths
        self.dense_depth_paths = dense_depth_paths
        self.sparse_depth_paths = sparse_depth_paths

        self.n_sample = len(image_paths)

        for paths in [dense_depth_paths, sparse_depth_paths]:
            assert len(paths) == self.n_sample

        self.use_image_triplet = use_image_triplet

        self.data_format = 'CHW'

    def __getitem__(self, index):
        # Load image
        if self.use_image_triplet:
            _, image, _ = load_image_triplet(
                self.image_paths[index],
                normalize=False)
        else:
            image = data_utils.load_image(
                self.image_paths[index],
                normalize=False,
                data_format=self.data_format)

        # Load sparse depth
        sparse_depth = data_utils.load_depth(
            self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load dense depth
        dense_depth = data_utils.load_depth(
            self.dense_depth_paths[index],
            data_format=self.data_format)

        # Convert to float32
        inputs = [
            T.astype(np.float32)
            for T in [image, dense_depth, sparse_depth]
        ]

        return inputs

    def __len__(self):
        return len(self.image_paths)


class FusionNetStandaloneTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image at time t-1, t, and t+1
        (2) sparse depth at time t
        (3) camera intrinsics matrix

    Arg(s):
        images_paths : list[str]
            paths to image triplets
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to 3 x 3 camera intrinsics matrix
        random_crop_shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
    '''

    def __init__(self,
                 images_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 random_crop_shape=None,
                 random_crop_type=['none']):

        self.images_paths = images_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths

        self.n_sample = len(images_paths)

        for paths in [sparse_depth_paths, intrinsics_paths]:
            assert len(paths) == self.n_sample

        self.random_crop_shape = random_crop_shape
        self.do_random_crop = \
            self.random_crop_shape is not None and all([x > 0 for x in self.random_crop_shape])

        # Augmentation
        self.random_crop_type = random_crop_type

        self.data_format = 'CHW'

    def __getitem__(self, index):
        # Load image at t-1, t, t+1
        image1, image0, image2 = load_image_triplet(
            self.images_paths[index],
            normalize=False,
            data_format=self.data_format)

        # Load sparse depth at time t
        sparse_depth0 = data_utils.load_depth(
            self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        intrinsics = np.load(self.intrinsics_paths[index]).astype(np.float32)

        inputs = [
            image0, image1, image2, sparse_depth0
        ]

        # Crop image, depth and adjust intrinsics
        if self.do_random_crop:
            inputs, [intrinsics] = random_crop(
                inputs=inputs,
                shape=self.random_crop_shape,
                intrinsics=[intrinsics],
                crop_type=self.random_crop_type)

        # Convert to float32
        inputs = inputs + [intrinsics]

        inputs = [
            T.astype(np.float32)
            for T in inputs
        ]

        return inputs

    def __len__(self):
        return self.n_sample


class FusionNetStandaloneInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) sparse depth

    Arg(s):
        image_paths : list[str]
            paths to image triplets
        sparse_depth_paths : list[str]
            paths to sparse depth maps
    '''

    def __init__(self,
                 image_paths,
                 sparse_depth_paths,
                 use_image_triplet=True):

        self.image_paths = image_paths
        self.sparse_depth_paths = sparse_depth_paths

        self.n_sample = len(image_paths)

        assert len(sparse_depth_paths) == self.n_sample

        self.use_image_triplet = use_image_triplet

        self.data_format = 'CHW'

    def __getitem__(self, index):
        # Load image
        if self.use_image_triplet:
            _, image, _ = load_image_triplet(
                self.image_paths[index],
                normalize=False)
        else:
            image = data_utils.load_image(
                self.image_paths[index],
                normalize=False,
                data_format=self.data_format)

        # Load sparse depth
        sparse_depth = data_utils.load_depth(
            self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Convert to float32
        inputs = [
            T.astype(np.float32)
            for T in [image, sparse_depth]
        ]

        return inputs

    def __len__(self):
        return len(self.image_paths)
