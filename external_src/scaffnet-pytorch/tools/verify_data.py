import sys, argparse
import numpy as np
import multiprocessing as mp

sys.path.insert(0, 'src')
import data_utils


def process_data(paths):
    '''
    Process image and depth map paths

    Arg(s):
        pathss : tuple
            list[str] : image paths
            list[str] : depth maps paths
            int : min_points_depth
    Returns:
        bool : True if data is valid else False
    '''

    image_paths, depth_paths, n_min_points_depth = paths

    image_valid = True
    depth_valid = True

    for image_path in image_paths:

        if image_path is not None:
            image = data_utils.load_image(image_path, normalize=False)

            image_max = np.max(image)
            image_min = np.min(image)

            if image_max > 255 or image_min < 0:
                image_valid = False

    for depth_path in depth_paths:

        if depth_path is not None:
            depth = data_utils.load_depth(depth_path)

            depth_max = np.max(depth)
            depth_min = np.min(depth)

            if depth_max > 255 or depth_min < 0:
                depth_valid = False

            validity_map = np.where(depth > 0.0, 1.0, 0.0)

            if np.sum(validity_map) < n_min_points_depth:
                depth_valid = False

    return image_valid and depth_valid


def delete_by_indices(list_object, indices):
    indices = sorted(indices, reverse=True)

    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Input filepaths
    parser.add_argument('--image_paths',
        nargs='+', type=str, default=None, help='List of paths to files containing image paths')
    parser.add_argument('--depth_paths',
        nargs='+', type=str, default=None, help='List of paths to files containing depth map paths')
    parser.add_argument('--n_min_points_depth',
        type=int, default=300, help='Number of points to expect in depth map')
    parser.add_argument('--n_thread',
        type=int, default=8, help='Number of threads to use')

    args = parser.parse_args()

    if args.image_paths is not None:
        image_paths = [
            data_utils.read_paths(path) for path in args.image_paths
        ]
    else:
        image_paths = [[]]

    if args.depth_paths is not None:
        depth_paths = [
            data_utils.read_paths(path) for path in args.depth_paths
        ]
    else:
        depth_paths = [[]]

    n_sample = max(len(image_paths[0]), len(depth_paths[0]))

    if len(image_paths) < n_sample:
        image_paths = [[None] * n_sample]

    if len(image_paths) < n_sample:
        image_paths = [[None] * n_sample]

    pool_inputs = []
    pool_input_image_paths = []
    pool_input_depth_paths = []

    for _image_paths in zip(*image_paths):
        pool_input_image_paths.append(_image_paths)

    for _depth_paths in zip(*depth_paths):
        pool_input_depth_paths.append(_depth_paths)

    for pool_input in zip(pool_input_image_paths, pool_input_depth_paths):
        pool_input = (*pool_input, args.n_min_points_depth)
        pool_inputs.append(pool_input)

    print('Processing {} paths...'.format(len(pool_inputs)))

    with mp.Pool(args.n_thread) as pool:
        pool_results = pool.map(process_data, pool_inputs)

    for pool_input_paths, is_valid in zip(pool_inputs, pool_results):

        if not is_valid:
            print('Found invalid paths:')
            for path in pool_input_paths[:2]:
                print(path)

    indices = [i for i, is_valid in enumerate(pool_results) if not is_valid]

    if args.image_paths is not None:
        for paths, filepath in zip(image_paths, args.image_paths):
            delete_by_indices(paths, indices)
            data_utils.write(filepath, paths)

    if args.depth_paths is not None:
        for paths, filepath in zip(depth_paths, args.depth_paths):
            delete_by_indices(paths, indices)
            data_utils.write_paths(filepath, paths)
