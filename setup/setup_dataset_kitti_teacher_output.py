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

import os, sys, argparse
import torch
sys.path.insert(0, 'src')
import datasets, data_utils
from external_model import ExternalModel
from run_external_model import run


'''
Input paths
'''
KITTI_RAW_DATA_DIRPATH = os.path.join('data', 'kitti_raw_data')
KITTI_DEPTH_COMPLETION_DIRPATH = os.path.join('data', 'kitti_depth_completion')

TRAIN_REF_DIRPATH = os.path.join('training', 'kitti')
VAL_REF_DIRPATH = os.path.join('validation', 'kitti')
TEST_REF_DIRPATH = os.path.join('testing', 'kitti')

TRAIN_IMAGES_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_images_left.txt')
TRAIN_IMAGES_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_images_right.txt')
TRAIN_SPARSE_DEPTH_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_sparse_depth_left.txt')
TRAIN_SPARSE_DEPTH_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_sparse_depth_right.txt')
TRAIN_INTRINSICS_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_intrinsics_left.txt')
TRAIN_INTRINSICS_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_intrinsics_right.txt')

TRAIN_NONSTATIC_SPARSE_DEPTH_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_nonstatic_sparse_depth_left.txt')
TRAIN_NONSTATIC_SPARSE_DEPTH_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_nonstatic_sparse_depth_right.txt')

UNUSED_IMAGE_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_unused_image_left.txt')
UNUSED_IMAGE_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_unused_image_right.txt')
UNUSED_SPARSE_DEPTH_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_unused_sparse_depth_left.txt')
UNUSED_SPARSE_DEPTH_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_unused_sparse_depth_right.txt')
UNUSED_INTRINSICS_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_unused_intrinsics_left.txt')
UNUSED_INTRINSICS_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_unused_intrinsics_right.txt')

VAL_IMAGE_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'kitti_val_image.txt')
VAL_SPARSE_DEPTH_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'kitti_val_sparse_depth.txt')
VAL_INTRINSICS_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'kitti_val_intrinsics.txt')

'''
Output paths
'''
KITTI_DEPTH_COMPLETION_DERIVED_DIRPATH = os.path.join(
    'data', 'kitti_depth_completion_mondi')

TRAIN_TEACHER_OUTPUT_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_teacher_output_left-{}.txt')
TRAIN_TEACHER_OUTPUT_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_teacher_output_right-{}.txt')

TRAIN_NONSTATIC_TEACHER_OUTPUT_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_nonstatic_teacher_output_left-{}.txt')
TRAIN_NONSTATIC_TEACHER_OUTPUT_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_nonstatic_teacher_output_right-{}.txt')

UNUSED_TEACHER_OUTPUT_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_unused_teacher_output_left-{}.txt')
UNUSED_TEACHER_OUTPUT_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_unused_teacher_output_right-{}.txt')

VAL_TEACHER_OUTPUT_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'kitti_val_teacher_output-{}.txt')


def setup_dataset_kitti_training(external_models,
                                 external_models_restore_paths,
                                 min_predict_depth,
                                 max_predict_depth,
                                 paths_only):
    '''
    Creates teacher_output based on external models

    Arg(s):
        external_models : list[str]
            name of external models to use
        external_models_restore_paths : list[str]
            paths to checkpoints for each external model
        min_predict_depth : float
            minimum value for predicted depth
        max_predict_depth : float
            maximum value for predicted depth
        paths_only : bool
            if set, then only produces paths
    '''

    assert len(external_models) == len(external_models_restore_paths), \
        'Length of external models list does not match length of restore paths list.'

    # Read input paths
    train_images_left_paths = data_utils.read_paths(TRAIN_IMAGES_LEFT_FILEPATH)
    train_images_right_paths = data_utils.read_paths(TRAIN_IMAGES_RIGHT_FILEPATH)
    train_sparse_depth_left_paths = data_utils.read_paths(TRAIN_SPARSE_DEPTH_LEFT_FILEPATH)
    train_sparse_depth_right_paths = data_utils.read_paths(TRAIN_SPARSE_DEPTH_RIGHT_FILEPATH)
    train_intrinsics_left_paths = data_utils.read_paths(TRAIN_INTRINSICS_LEFT_FILEPATH)
    train_intrinsics_right_paths = data_utils.read_paths(TRAIN_INTRINSICS_RIGHT_FILEPATH)

    train_nonstatic_sparse_depth_left_paths = data_utils.read_paths(TRAIN_NONSTATIC_SPARSE_DEPTH_LEFT_FILEPATH)
    train_nonstatic_sparse_depth_right_paths = data_utils.read_paths(TRAIN_NONSTATIC_SPARSE_DEPTH_RIGHT_FILEPATH)

    unused_image_left_paths = data_utils.read_paths(UNUSED_IMAGE_LEFT_FILEPATH)
    unused_image_right_paths = data_utils.read_paths(UNUSED_IMAGE_RIGHT_FILEPATH)
    unused_sparse_depth_left_paths = data_utils.read_paths(UNUSED_SPARSE_DEPTH_LEFT_FILEPATH)
    unused_sparse_depth_right_paths = data_utils.read_paths(UNUSED_SPARSE_DEPTH_RIGHT_FILEPATH)
    unused_intrinsics_left_paths = data_utils.read_paths(UNUSED_INTRINSICS_LEFT_FILEPATH)
    unused_intrinsics_right_paths = data_utils.read_paths(UNUSED_INTRINSICS_RIGHT_FILEPATH)

    val_image_paths = data_utils.read_paths(VAL_IMAGE_FILEPATH)
    val_sparse_depth_paths = data_utils.read_paths(VAL_SPARSE_DEPTH_FILEPATH)
    val_intrinsics_paths = data_utils.read_paths(VAL_INTRINSICS_FILEPATH)

    # Preallocate output paths
    train_teacher_output_left_paths = [
        sparse_depth_path \
            .replace(KITTI_DEPTH_COMPLETION_DIRPATH, KITTI_DEPTH_COMPLETION_DERIVED_DIRPATH) \
            .replace('sparse_depth', os.path.join('teacher_output', '{}'))
        for sparse_depth_path in train_sparse_depth_left_paths
    ]

    train_teacher_output_right_paths = [
        sparse_depth_path \
            .replace(KITTI_DEPTH_COMPLETION_DIRPATH, KITTI_DEPTH_COMPLETION_DERIVED_DIRPATH) \
            .replace('sparse_depth', os.path.join('teacher_output', '{}'))
        for sparse_depth_path in train_sparse_depth_right_paths
    ]

    train_nonstatic_teacher_output_left_paths = [
        sparse_depth_path \
            .replace(KITTI_DEPTH_COMPLETION_DIRPATH, KITTI_DEPTH_COMPLETION_DERIVED_DIRPATH) \
            .replace('sparse_depth', os.path.join('teacher_output', '{}'))
        for sparse_depth_path in train_nonstatic_sparse_depth_left_paths
    ]

    train_nonstatic_teacher_output_right_paths = [
        sparse_depth_path \
            .replace(KITTI_DEPTH_COMPLETION_DIRPATH, KITTI_DEPTH_COMPLETION_DERIVED_DIRPATH) \
            .replace('sparse_depth', os.path.join('teacher_output', '{}'))
        for sparse_depth_path in train_nonstatic_sparse_depth_right_paths
    ]

    unused_teacher_output_left_paths = [
        sparse_depth_path \
            .replace(KITTI_DEPTH_COMPLETION_DIRPATH, KITTI_DEPTH_COMPLETION_DERIVED_DIRPATH) \
            .replace('sparse_depth', os.path.join('teacher_output', '{}'))
        for sparse_depth_path in unused_sparse_depth_left_paths
    ]

    unused_teacher_output_right_paths = [
        sparse_depth_path \
            .replace(KITTI_DEPTH_COMPLETION_DIRPATH, KITTI_DEPTH_COMPLETION_DERIVED_DIRPATH) \
            .replace('sparse_depth', os.path.join('teacher_output', '{}'))
        for sparse_depth_path in unused_sparse_depth_right_paths
    ]

    val_teacher_output_paths = [
        sparse_depth_path \
            .replace(KITTI_DEPTH_COMPLETION_DIRPATH, KITTI_DEPTH_COMPLETION_DERIVED_DIRPATH) \
            .replace('sparse_depth', os.path.join('teacher_output', '{}'))
        for sparse_depth_path in val_sparse_depth_paths
    ]

    # Combine left and right camera paths to a single list
    train_image_paths = \
        train_images_left_paths + train_images_right_paths
    train_sparse_depth_paths = \
        train_sparse_depth_left_paths + train_sparse_depth_right_paths
    train_intrinsics_paths = \
        train_intrinsics_left_paths + train_intrinsics_right_paths
    train_teacher_output_paths = \
        train_teacher_output_left_paths + train_teacher_output_right_paths

    unused_image_paths = \
        unused_image_left_paths + unused_image_right_paths
    unused_sparse_depth_paths = \
        unused_sparse_depth_left_paths + unused_sparse_depth_right_paths
    unused_intrinsics_paths = \
        unused_intrinsics_left_paths + unused_intrinsics_right_paths
    unused_teacher_output_paths = \
        unused_teacher_output_left_paths + unused_teacher_output_right_paths

    input_paths = [
        [
            'training',
            [
                train_image_paths,
                train_sparse_depth_paths,
                train_intrinsics_paths,
                train_teacher_output_paths
            ]
        ],
        [
            'unused',
            [
                unused_image_paths,
                unused_sparse_depth_paths,
                unused_intrinsics_paths,
                unused_teacher_output_paths
            ]
        ],
        [
            'val',
            [
                val_image_paths,
                val_sparse_depth_paths,
                val_intrinsics_paths,
                val_teacher_output_paths
            ]
        ]
    ]

    # Run external model to produce teacher_output
    for model_name, restore_path in zip(external_models, external_models_restore_paths):

        # Build external depth completion network
        model = ExternalModel(
            model_name=model_name,
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth,
            device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # Restore model and set to evaluation mode
        try:
            model.restore_model(restore_path)
        except Exception:
            model.data_parallel()
            model.restore_model(restore_path)

        model.eval()

        print('Generating teacher_output using {} model'.format(model_name))
        print('Restoring from {}'.format(restore_path))

        for tag, paths in input_paths:

            image_paths, \
                sparse_depth_paths, \
                intrinsics_paths, \
                teacher_output_paths = paths

            for p in paths:
                assert len(image_paths) == len(p)

            output_paths = [
                teacher_output_path.format(model_name)
                for teacher_output_path in teacher_output_paths
            ]

            # Create output directories
            output_dirpaths = set([os.path.dirname(path) for path in output_paths])

            for dirpath in output_dirpaths:
                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)

            n_sample = len(image_paths)

            # Instantiate dataloader
            dataloader = torch.utils.data.DataLoader(
                datasets.DepthCompletionInferenceDataset(
                    image_paths=image_paths,
                    sparse_depth_paths=sparse_depth_paths,
                    intrinsics_paths=intrinsics_paths,
                    load_image_triplets=True if tag == 'training' else False),
                batch_size=1,
                shuffle=False,
                num_workers=1,
                drop_last=False)

            if not paths_only:
                print('Generating teacher_output for {} {} samples'.format(n_sample, tag))

                # Write to teacher_output to disk
                run(model, dataloader, output_paths=output_paths, verbose=True)

    # Write paths to disk
    train_filepaths = [
        (train_teacher_output_left_paths, TRAIN_TEACHER_OUTPUT_LEFT_FILEPATH),
        (train_teacher_output_right_paths, TRAIN_TEACHER_OUTPUT_RIGHT_FILEPATH),
        (train_nonstatic_teacher_output_left_paths, TRAIN_NONSTATIC_TEACHER_OUTPUT_LEFT_FILEPATH),
        (train_nonstatic_teacher_output_right_paths, TRAIN_NONSTATIC_TEACHER_OUTPUT_RIGHT_FILEPATH)
    ]

    unused_filepaths = [
        (unused_teacher_output_left_paths, UNUSED_TEACHER_OUTPUT_LEFT_FILEPATH),
        (unused_teacher_output_right_paths, UNUSED_TEACHER_OUTPUT_RIGHT_FILEPATH),
        (None, None),
        (None, None)
    ]

    val_filepaths = [
        (val_teacher_output_paths, VAL_TEACHER_OUTPUT_FILEPATH),
        (val_teacher_output_paths, VAL_TEACHER_OUTPUT_FILEPATH),
        (None, None),
        (None, None)
    ]

    tags = ['training', 'unused', 'val']
    data_filepaths = [train_filepaths, unused_filepaths, val_filepaths]

    for tag, filepaths in zip(tags, data_filepaths):

        (teacher_output_left_paths, teacher_output_left_filepath), \
            (teacher_output_right_paths, teacher_output_right_filepath), \
            (nonstatic_teacher_output_left_paths, nonstatic_teacher_output_left_filepath), \
            (nonstatic_teacher_output_right_paths, nonstatic_teacher_output_right_filepath) = filepaths

        for model_name in external_models:

            # Add model name as a tag to output paths
            model_teacher_output_left_paths = [
                teacher_output_left_path.format(model_name)
                for teacher_output_left_path in teacher_output_left_paths
            ]

            model_teacher_output_right_paths = [
                teacher_output_right_path.format(model_name)
                for teacher_output_right_path in teacher_output_right_paths
            ]

            model_teacher_output_left_filepath = \
                teacher_output_left_filepath.format(model_name)

            model_teacher_output_right_filepath = \
                teacher_output_right_filepath.format(model_name)

            # Write to file
            if tag == 'training' or tag == 'unused':
                print('Storing {} {} left {} teacher_output file paths into: {}'.format(
                    len(model_teacher_output_left_paths),
                    model_name,
                    tag,
                    model_teacher_output_left_filepath))
                data_utils.write_paths(
                    model_teacher_output_left_filepath,
                    model_teacher_output_left_paths)

                print('Storing {} {} right {} teacher_output file paths into: {}'.format(
                    len(model_teacher_output_right_paths),
                    model_name,
                    tag,
                    model_teacher_output_right_filepath))
                data_utils.write_paths(
                    model_teacher_output_right_filepath,
                    model_teacher_output_right_paths)
            else:
                print('Storing {} {} {} teacher_output file paths into: {}'.format(
                    len(model_teacher_output_left_paths),
                    model_name,
                    tag,
                    model_teacher_output_left_filepath))
                data_utils.write_paths(
                    model_teacher_output_left_filepath,
                    model_teacher_output_left_paths)

            if tag == 'training':
                # Create paths for teacher_output nonstatic predictions
                model_nonstatic_teacher_output_left_paths = [
                    nonstatic_teacher_output_left_path.format(model_name)
                    for nonstatic_teacher_output_left_path in nonstatic_teacher_output_left_paths
                ]

                model_nonstatic_teacher_output_right_paths = [
                    nonstatic_teacher_output_right_path.format(model_name)
                    for nonstatic_teacher_output_right_path in nonstatic_teacher_output_right_paths
                ]

                model_nonstatic_teacher_output_left_filepath = \
                    nonstatic_teacher_output_left_filepath.format(model_name)

                model_nonstatic_teacher_output_right_filepath = \
                    nonstatic_teacher_output_right_filepath.format(model_name)

                # Write to file
                print('Storing {} nonstatic {} left {} teacher_output file paths into: {}'.format(
                    len(model_nonstatic_teacher_output_left_paths),
                    model_name,
                    tag,
                    model_nonstatic_teacher_output_left_filepath))
                data_utils.write_paths(
                    model_nonstatic_teacher_output_left_filepath,
                    model_nonstatic_teacher_output_left_paths)

                print('Storing {} nonstatic {} right {} teacher_output file paths into: {}'.format(
                    len(model_nonstatic_teacher_output_right_paths),
                    model_name,
                    tag,
                    model_nonstatic_teacher_output_right_filepath))
                data_utils.write_paths(
                    model_nonstatic_teacher_output_right_filepath,
                    model_nonstatic_teacher_output_right_paths)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--external_models',
        nargs='+', type=str, required=True, help='Space delimited list of external model to use for teacher outputs')
    parser.add_argument('--external_models_restore_paths',
        nargs='+', type=str, required=True, help='Space delimited list of checkpoint paths for external models ')
    parser.add_argument('--min_predict_depth',
        type=float, default=1.5, help='Minimum value for predicted depth')
    parser.add_argument('--max_predict_depth',
        type=float, default=100.0, help='Maximum value for predicted depth')
    parser.add_argument('--paths_only',
        action='store_true', help='If set, then generate paths only')

    args = parser.parse_args()

    # Create directories for output files
    for dirpath in [TRAIN_REF_DIRPATH, VAL_REF_DIRPATH, TEST_REF_DIRPATH]:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    # Set up dataset
    setup_dataset_kitti_training(
        args.external_models,
        args.external_models_restore_paths,
        args.min_predict_depth,
        args.max_predict_depth,
        args.paths_only)
