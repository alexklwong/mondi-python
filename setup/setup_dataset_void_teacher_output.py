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
VOID_RELEASE_DATA_DIRPATH = os.path.join('data', 'void_release', 'void_1500', 'data')

TRAIN_REF_DIRPATH = os.path.join('training', 'void')

TRAIN_IMAGE_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'void_train_image_1500.txt'
)
TRAIN_SPARSE_DEPTH_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'void_train_sparse_depth_1500.txt'
)
TRAIN_INTRINSICS_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'void_train_intrinsics_1500.txt'
)

'''
Output paths
'''
VOID_DEPTH_COMPLETION_OUTPUT_DIRPATH = os.path.join(
    'data', 'void_mondi', 'void_1500', 'teacher_output')

TRAIN_TEACHER_OUTPUT_1500_OUTPUT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'void_train_teacher_output_1500-{}.txt')


def setup_dataset_void_training(external_models,
                                external_models_restore_paths,
                                min_predict_depth,
                                max_predict_depth,
                                paths_only):
    '''
    Creates teacher output based on external models

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
    train_image_paths = data_utils.read_paths(TRAIN_IMAGE_FILEPATH)
    train_sparse_depth_paths = data_utils.read_paths(TRAIN_SPARSE_DEPTH_FILEPATH)
    train_intrinsics_paths = data_utils.read_paths(TRAIN_INTRINSICS_FILEPATH)

    # Preallocate output paths
    train_teacher_output_paths = [
        sparse_depth_path \
            .replace(VOID_RELEASE_DATA_DIRPATH, os.path.join(VOID_DEPTH_COMPLETION_OUTPUT_DIRPATH, '{}')) \
            .replace('/sparse_depth', '')
        for sparse_depth_path in train_sparse_depth_paths
    ]

    input_paths = [
        [
            'training',
            [
                train_image_paths,
                train_sparse_depth_paths,
                train_intrinsics_paths,
                train_teacher_output_paths
            ]
        ]
    ]

    # Run external model to produce teacher output
    for model_name, restore_path in zip(external_models, external_models_restore_paths):

        # Build external depth completion network
        model = ExternalModel(
            model_name=model_name,
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth,
            device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),)

        # Restore model and set to evaluation mode
        try:
            model.restore_model(restore_path)
        except Exception:
            model.data_parallel()
            model.restore_model(restore_path)

        model.eval()

        print('Generating teacher output using {} model'.format(model_name))
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
                print('Generating teacher output for {} {} samples'.format(n_sample, tag))

                # Write to teacher output to disk
                run(model, dataloader, output_paths=output_paths, verbose=True)

    # Write paths to disk
    train_filepaths = [
        (train_teacher_output_paths, TRAIN_TEACHER_OUTPUT_1500_OUTPUT_FILEPATH)
    ]

    for tag, filepaths in zip(['training'], [train_filepaths]):

        [(teacher_output_paths, teacher_output_filepath)] = filepaths

        for model_name in external_models:

            # Add model name as a tag to output paths
            model_teacher_output_paths = [
                teacher_output_path.format(model_name)
                for teacher_output_path in teacher_output_paths
            ]

            model_teacher_output_filepath = \
                teacher_output_filepath.format(model_name)
            # Write to file
            print('Storing {} {} {} teacher output file paths into: {}'.format(
                len(model_teacher_output_paths),
                model_name,
                tag,
                model_teacher_output_filepath))
            data_utils.write_paths(
                model_teacher_output_filepath,
                model_teacher_output_paths)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--external_models',
        nargs='+', type=str, required=True, help='Space delimited list of external model to use for teacher output')
    parser.add_argument('--external_models_restore_paths',
        nargs='+', type=str, required=True, help='Space delimited list of checkpoint paths for external models')
    parser.add_argument('--min_predict_depth',
        type=float, default=0.1, help='Minimum value for predicted depth')
    parser.add_argument('--max_predict_depth',
        type=float, default=8.0, help='Maximum value for predicted depth')
    parser.add_argument('--paths_only',
        action='store_true', help='If set, then generate paths only')

    args = parser.parse_args()

    # Create directories for output files
    for dirpath in [TRAIN_REF_DIRPATH, VOID_DEPTH_COMPLETION_OUTPUT_DIRPATH]:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    # Set up dataset
    setup_dataset_void_training(
        args.external_models,
        args.external_models_restore_paths,
        args.min_predict_depth,
        args.max_predict_depth,
        args.paths_only)
