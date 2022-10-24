import sys, os, glob, cv2
import numpy as np
import multiprocessing as mp
from skimage import morphology as skmorph
sys.path.insert(0, 'src')
import data_utils


N_PER_FRAME = 1
STEP_SIZE = 1


'''
Paths for KITTI dataset
'''
KITTI_ROOT_DIRPATH = os.path.join('data', 'kitti_depth_completion')
KITTI_TRAIN_SPARSE_DEPTH_DIRPATH = os.path.join(
    KITTI_ROOT_DIRPATH, 'train_val_split', 'sparse_depth', 'train')
# To be concatenated to sequence path
KITTI_SPARSE_DEPTH_REFPATH = os.path.join('proj_depth', 'velodyne_raw')


'''
Paths for Virtual KITTI dataset
'''
VKITTI_ROOT_DIRPATH = os.path.join('data', 'virtual_kitti')
VKITTI_TRAIN_DEPTH_REFPATH = 'vkitti_2.0.3_depth'
# Note: we only need to use the clone directory since lighting change only affects RGB
VKITTI_TRAIN_DEPTH_DIRPATH = os.path.join(VKITTI_ROOT_DIRPATH, VKITTI_TRAIN_DEPTH_REFPATH)


'''
Output directory
'''
OUTPUT_ROOT_DIRPATH = os.path.join('data', 'virtual_kitti_scaffnet')
OUTPUT_REF_DIRPATH = 'training'
OUTPUT_SPARSE_DEPTH_FILEPATH = os.path.join(OUTPUT_REF_DIRPATH, 'vkitti_train_sparse_depth.txt')
OUTPUT_VALIDITY_MAP_FILEPATH = os.path.join(OUTPUT_REF_DIRPATH, 'vkitti_train_validity_map.txt')
OUTPUT_SEMI_DENSE_DEPTH_FILEPATH = os.path.join(OUTPUT_REF_DIRPATH, 'vkitti_train_semi_dense_depth.txt')
OUTPUT_DENSE_DEPTH_FILEPATH = os.path.join(OUTPUT_REF_DIRPATH, 'vkitti_train_dense_depth.txt')
OUTPUT_GROUND_TRUTH_FILEPATH = os.path.join(OUTPUT_REF_DIRPATH, 'vkitti_train_ground_truth.txt')


def process_frame(args):
    vkitti_ground_truth_path, kitti_sparse_depth_paths, output_dirpaths = args

    # Load virtual KITTI groundtruth depth
    vkitti_ground_truth = data_utils.load_depth(vkitti_ground_truth_path, multiplier=100.0)

    # Set up filename and output paths
    filename = os.path.basename(vkitti_ground_truth_path)
    filename, ext = os.path.splitext(filename)

    vkitti_sparse_depth_dirpath, vkitti_validity_map_dirpath, \
        vkitti_semi_dense_depth_dirpath, vkitti_dense_depth_dirpath, \
        vkitti_ground_truth_dirpath = output_dirpaths

    # Output filepaths
    vkitti_sparse_depth_paths = []
    vkitti_validity_map_paths = []
    vkitti_semi_dense_depth_paths = []
    vkitti_dense_depth_paths = []
    vkitti_ground_truth_paths = \
        [os.path.join(vkitti_ground_truth_dirpath, filename + ext)] * len(kitti_sparse_depth_paths)

    for idx, kitti_sparse_depth_path in enumerate(kitti_sparse_depth_paths):
        # Load KITTI validity map
        _, kitti_validity_map = data_utils.load_depth_with_validity_map(kitti_sparse_depth_path)

        # Resize groundtruth to size of validity map
        if kitti_validity_map.shape != vkitti_ground_truth.shape:
            vkitti_ground_truth = cv2.resize(vkitti_ground_truth,
                dsize=(kitti_validity_map.shape[1], kitti_validity_map.shape[0]),
                interpolation=cv2.INTER_NEAREST)

        # Get Virtual KITTI sparse and dense depth without sky
        vkitti_validity_map = np.ones_like(kitti_validity_map)
        vkitti_validity_map[vkitti_ground_truth > 255.0] = 0.0
        vkitti_dense_depth = vkitti_ground_truth * vkitti_validity_map
        vkitti_sparse_depth = vkitti_dense_depth * kitti_validity_map
        vkitti_semi_dense_depth = \
            vkitti_dense_depth * np.where(skmorph.convex_hull_image(kitti_validity_map), 1, 0)

        # Append index to filename
        output_filename = filename + '-{}'.format(idx) + ext

        # Store output paths
        vkitti_sparse_depth_paths.append(os.path.join(vkitti_sparse_depth_dirpath, output_filename))
        vkitti_validity_map_paths.append(os.path.join(vkitti_validity_map_dirpath, output_filename))
        vkitti_semi_dense_depth_paths.append(os.path.join(vkitti_semi_dense_depth_dirpath, output_filename))
        vkitti_dense_depth_paths.append(os.path.join(vkitti_dense_depth_dirpath, output_filename))

        # Save to as PNG to disk
        data_utils.save_depth(vkitti_sparse_depth, vkitti_sparse_depth_paths[-1])
        data_utils.save_validity_map(kitti_validity_map, vkitti_validity_map_paths[-1])
        data_utils.save_depth(vkitti_semi_dense_depth, vkitti_semi_dense_depth_paths[-1])
        data_utils.save_depth(vkitti_dense_depth, vkitti_dense_depth_paths[-1])

        # Only save groundtruth depth once
        if idx == 0:
            data_utils.save_depth(vkitti_ground_truth, vkitti_ground_truth_paths[-1])

    return (vkitti_sparse_depth_paths, vkitti_validity_map_paths,
        vkitti_semi_dense_depth_paths, vkitti_dense_depth_paths, vkitti_ground_truth_paths)


'''
Generate sparse, semi-dense, dense depth with validity map
'''
if not os.path.exists(OUTPUT_REF_DIRPATH):
    os.makedirs(OUTPUT_REF_DIRPATH)

# Obtain the set of sequence dirpaths
kitti_sequence_dirpaths = glob.glob(os.path.join(KITTI_TRAIN_SPARSE_DEPTH_DIRPATH, '*/'))
vkitti_sequence_dirpaths = glob.glob(os.path.join(VKITTI_TRAIN_DEPTH_DIRPATH, '*/'))

output_sparse_depth_paths = []
output_validity_map_paths = []
output_semi_dense_depth_paths = []
output_dense_depth_paths = []
output_ground_truth_paths = []
for vkitti_sequence_dirpath in vkitti_sequence_dirpaths:
    print('Processing Virtual KITTI sequence: {}'.format(vkitti_sequence_dirpath))

    # Select Virtual KITTI sequence: data/virtual_kitti/vkitti_2.0.3_depth/Scene01/clone/frames/depth/
    vkitti_sequence_dirpath = os.path.join(vkitti_sequence_dirpath, 'clone', 'frames', 'depth')

    # Construct output directory: data/virtual_kitti_scaffnet/vkitti_2.0.3_depth/Scene01/clone/frames/depth/
    output_sequence_dirpath = vkitti_sequence_dirpath.replace(VKITTI_ROOT_DIRPATH, OUTPUT_ROOT_DIRPATH)

    n_output = 0
    for kitti_sequence_dirpath in kitti_sequence_dirpaths:
        # Select KITTI sequence, since it is a directory last element is empty so grab the second til last
        kitti_sequence = kitti_sequence_dirpath.split(os.sep)[-2]
        kitti_sequence_dirpath = os.path.join(kitti_sequence_dirpath, KITTI_SPARSE_DEPTH_REFPATH)

        camera_dirpaths = zip(['image_02', 'image_03'], ['Camera_0', 'Camera_1'])

        for kitti_camera_dirpath, vkitti_camera_dirpath in camera_dirpaths:
            kitti_sequence_filepaths = sorted(glob.glob(
                os.path.join(kitti_sequence_dirpath, kitti_camera_dirpath, '*.png')))

            vkitti_sequence_filepaths = sorted(glob.glob(
                os.path.join(vkitti_sequence_dirpath, vkitti_camera_dirpath, '*.png')))

            # Construct output paths
            output_sparse_depth_dirpath = os.path.join(
                output_sequence_dirpath, kitti_sequence, vkitti_camera_dirpath, 'sparse')
            output_validity_map_dirpath = os.path.join(
                output_sequence_dirpath, kitti_sequence, vkitti_camera_dirpath, 'validity_map')
            output_semi_dense_depth_dirpath = os.path.join(
                output_sequence_dirpath, kitti_sequence, vkitti_camera_dirpath, 'semi_dense')
            output_dense_depth_dirpath = os.path.join(
                output_sequence_dirpath, kitti_sequence, vkitti_camera_dirpath, 'dense')
            output_ground_truth_dirpath = os.path.join(
                output_sequence_dirpath, kitti_sequence, vkitti_camera_dirpath, 'ground_truth')

            output_dirpaths = [
                output_sparse_depth_dirpath,
                output_validity_map_dirpath,
                output_semi_dense_depth_dirpath,
                output_dense_depth_dirpath,
                output_ground_truth_dirpath
            ]

            for output_dirpath in output_dirpaths:
                if not os.path.exists(output_dirpath):
                    os.makedirs(output_dirpath)

            sufficient_samples = False

            pool_inputs = []

            if len(kitti_sequence_filepaths) + STEP_SIZE * N_PER_FRAME > len(vkitti_sequence_filepaths):
                sufficient_samples = True

            for vkitti_idx, vkitti_ground_truth_path in enumerate(vkitti_sequence_filepaths):
                start_idx = vkitti_idx if sufficient_samples else 0
                end_idx = start_idx + STEP_SIZE * N_PER_FRAME

                kitti_sparse_depth_paths = kitti_sequence_filepaths[start_idx:end_idx:STEP_SIZE]

                pool_inputs.append(
                    (vkitti_ground_truth_path, kitti_sparse_depth_paths, output_dirpaths))

            with mp.Pool() as pool:
                pool_results = pool.map(process_frame, pool_inputs)

            for result in pool_results:
                sparse_depth_paths, validity_map_paths, \
                    semi_dense_depth_paths, dense_depth_paths, ground_truth_paths = result

                n_output = n_output + len(sparse_depth_paths)

                output_sparse_depth_paths.extend(sparse_depth_paths)
                output_validity_map_paths.extend(validity_map_paths)
                output_semi_dense_depth_paths.extend(semi_dense_depth_paths)
                output_dense_depth_paths.extend(dense_depth_paths)
                output_ground_truth_paths.extend(ground_truth_paths)

    print('Generated {} total samples for {}'.format(n_output, vkitti_sequence_dirpath))


'''
Write paths to disk
'''
print('Writing {} sparse depth paths to {}'.format(
    len(output_sparse_depth_paths), OUTPUT_SPARSE_DEPTH_FILEPATH))
data_utils.write_paths(OUTPUT_SPARSE_DEPTH_FILEPATH, output_sparse_depth_paths)

print('Writing {} validity map paths to {}'.format(
    len(output_validity_map_paths), OUTPUT_VALIDITY_MAP_FILEPATH))
data_utils.write_paths(OUTPUT_VALIDITY_MAP_FILEPATH, output_validity_map_paths)

print('Writing {} semi-dense depth paths to {}'.format(
    len(output_semi_dense_depth_paths), OUTPUT_SEMI_DENSE_DEPTH_FILEPATH))
data_utils.write_paths(OUTPUT_SEMI_DENSE_DEPTH_FILEPATH, output_semi_dense_depth_paths)

print('Writing {} dense depth paths to {}'.format(
    len(output_dense_depth_paths), OUTPUT_DENSE_DEPTH_FILEPATH))
data_utils.write_paths(OUTPUT_DENSE_DEPTH_FILEPATH, output_dense_depth_paths)

print('Writing {} groundtruth depth paths to {}'.format(
    len(output_ground_truth_paths), OUTPUT_GROUND_TRUTH_FILEPATH))
data_utils.write_paths(OUTPUT_GROUND_TRUTH_FILEPATH, output_ground_truth_paths)
