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

import numpy as np
from PIL import Image
from skimage.restoration import inpaint
from scipy.interpolate import LinearNDInterpolator


def read_paths(filepath):
    '''
    Reads a newline delimited file containing paths

    Arg(s):
        filepath : str
            path to file to be read
    Return:
        list[str] : list of paths
    '''

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip('\n')

            # If there was nothing to read
            if path == '':
                break

            path_list.append(path)

    return path_list

def write_paths(filepath, paths):
    '''
    Stores line delimited paths into file

    Arg(s):
        filepath : str
            path to file to save paths
        paths : list[str]
            paths to write into file
    '''

    with open(filepath, 'w') as o:
        for idx in range(len(paths)):
            o.write(paths[idx] + '\n')

def load_image(path, normalize=True, data_format='HWC'):
    '''
    Loads an RGB image

    Arg(s):
        path : str
            path to RGB image
        normalize : bool
            if set, then normalize image between [0, 1]
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : H x W x C or C x H x W image
    '''

    # Load image
    image = Image.open(path).convert('RGB')

    # Convert to numpy
    image = np.asarray(image, np.float32)

    if data_format == 'HWC':
        pass
    elif data_format == 'CHW':
        image = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    # Normalize
    image = image / 255.0 if normalize else image

    return image

def load_depth_with_validity_map(path, data_format='HW'):
    '''
    Loads a depth map and validity map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : depth map
        numpy[float32] : binary validity map for available depth measurement locations
    '''

    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)

    # Assert 16-bit (not 8-bit) depth map
    z = z / 256.0
    z[z <= 0] = 0.0
    v = z.astype(np.float32)
    v[z > 0] = 1.0

    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        z = np.expand_dims(z, axis=0)
        v = np.expand_dims(v, axis=0)
    elif data_format == 'HWC':
        z = np.expand_dims(z, axis=-1)
        v = np.expand_dims(v, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return z, v

def load_depth(path, data_format='HW'):
    '''
    Loads a depth map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : depth map
    '''

    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)

    # Assert 16-bit (not 8-bit) depth map
    z = z / 256.0
    z[z <= 0] = 0.0

    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        z = np.expand_dims(z, axis=0)
    elif data_format == 'HWC':
        z = np.expand_dims(z, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return z

def save_depth(z, path):
    '''
    Saves a depth map to a 16-bit PNG file

    Arg(s):
        z : numpy[float32]
            depth map
        path : str
            path to store depth map
    '''

    z = np.uint32(z * 256.0)
    z = Image.fromarray(z, mode='I')
    z.save(path)

def load_validity_map(path, data_format='HW'):
    '''
    Loads a validity map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : binary validity map for available depth measurement locations
    '''

    # Loads validity map from 16-bit PNG file
    v = np.array(Image.open(path), dtype=np.float32)

    assert np.all(np.unique(v) == [0, 256])
    v[v > 0] = 1

    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        v = np.expand_dims(v, axis=0)
    elif data_format == 'HWC':
        v = np.expand_dims(v, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return v

def save_validity_map(v, path):
    '''
    Saves a validity map to a 16-bit PNG file

    Arg(s):
        v : numpy[float32]
            validity map
        path : str
            path to store validity map
    '''

    v[v <= 0] = 0.0
    v[v > 0] = 1.0
    v = np.uint32(v * 256.0)
    v = Image.fromarray(v, mode='I')
    v.save(path)

def load_calibration(path):
    '''
    Loads the calibration matrices for each camera (KITTI) and stores it as map

    Arg(s):
        path : str
            path to file to be read
    Returns:
        dict[str, float32] : map containing camera intrinsics keyed by camera id
    '''

    float_chars = set("0123456789.e+- ")
    data = {}

    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                try:
                    data[key] = np.asarray(
                        [float(x) for x in value.split(' ')])
                except ValueError:
                    pass
    return data

def inpainting(depth_map):
    '''
    Fill in almost dense depth_map with mask using Scikit Learn Image's inpaint_biharmonic
    Used for post processing depth maps

    Arg(s):
        depth_map : numpy[float32]
            N x 1 x H x W almost dense depth map
    Returns:
        numpy[float32] : N x 1 x H x W inpainted depth map
    '''

    # Create a mask from the depth map (1 represents holes, 0 represents background)
    n_batch = depth_map.shape[0]
    mask = np.where(depth_map == 0, 1, 0)

    if (mask == np.zeros_like(mask)).all():
        return depth_map

    # Perform inpainting with biharmonic interpolation
    for i in range(n_batch):
        depth = depth_map[i, ...]
        mask = np.where(depth == 0, 1, 0)
        if (mask == np.zeros_like(mask)).all():
            pass
        depth_map[i] = inpaint.inpaint_biharmonic(depth, mask)

    return depth_map


def interpolate_depth(depth_map, validity_map, log_space=False):
    '''
    Interpolate sparse depth with barycentric coordinates

    Arg(s):
        depth_map : numpy[float32]
            H x W depth map
        validity_map : numpy[float32]
            H x W depth map
        log_space : bool
            if set then produce in log space
    Returns:
        numpy[float32] : H x W interpolated depth map
    '''

    assert depth_map.ndim == 2 and validity_map.ndim == 2

    rows, cols = depth_map.shape
    data_row_idx, data_col_idx = np.where(validity_map)
    depth_values = depth_map[data_row_idx, data_col_idx]

    # Perform linear interpolation in log space
    if log_space:
        depth_values = np.log(depth_values)

    interpolator = LinearNDInterpolator(
        # points=Delaunay(np.stack([data_row_idx, data_col_idx], axis=1).astype(np.float32)),
        points=np.stack([data_row_idx, data_col_idx], axis=1),
        values=depth_values,
        fill_value=0 if not log_space else np.log(1e-3))

    query_row_idx, query_col_idx = np.meshgrid(
        np.arange(rows), np.arange(cols), indexing='ij')

    query_coord = np.stack(
        [query_row_idx.ravel(), query_col_idx.ravel()], axis=1)

    Z = interpolator(query_coord).reshape([rows, cols])

    if log_space:
        Z = np.exp(Z)
        Z[Z < 1e-1] = 0.0

    return Z
