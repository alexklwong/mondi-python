import os, sys, argparse
import numpy as np
import torch
sys.path.insert(0, 'src')
import data_utils, datasets
from scaffnet_model import ScaffNetModel


'''
Data directories
'''
VOID_DATA_DIRPATH = os.path.join(
    'data', 'void_release')

VOID_DATA_DERIVED_DIRPATH = os.path.join(
    'data', 'void_scaffnet')

TRAIN_REFS_DIRPATH      = os.path.join('training', 'void')
TEST_REFS_DIRPATH       = os.path.join('testing', 'void')


'''
Input file paths
'''
VOID_TRAIN_SPARSE_DEPTH_150_FILEPATH = \
    os.path.join(TRAIN_REFS_DIRPATH, 'void_train_sparse_depth_150.txt')
VOID_TRAIN_SPARSE_DEPTH_500_FILEPATH = \
    os.path.join(TRAIN_REFS_DIRPATH, 'void_train_sparse_depth_500.txt')
VOID_TRAIN_SPARSE_DEPTH_1500_FILEPATH = \
    os.path.join(TRAIN_REFS_DIRPATH, 'void_train_sparse_depth_1500.txt')

VOID_TEST_SPARSE_DEPTH_150_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_sparse_depth_150.txt')
VOID_TEST_SPARSE_DEPTH_500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_sparse_depth_500.txt')
VOID_TEST_SPARSE_DEPTH_1500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_sparse_depth_1500.txt')


'''
Output file paths
'''
VOID_TRAIN_PREDICT_DEPTH_150_FILEPATH = \
    os.path.join(TRAIN_REFS_DIRPATH, 'void_train_predict_depth_150.txt')
VOID_TRAIN_PREDICT_DEPTH_500_FILEPATH = \
    os.path.join(TRAIN_REFS_DIRPATH, 'void_train_predict_depth_500.txt')
VOID_TRAIN_PREDICT_DEPTH_1500_FILEPATH = \
    os.path.join(TRAIN_REFS_DIRPATH, 'void_train_predict_depth_1500.txt')

VOID_TEST_PREDICT_DEPTH_150_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_predict_depth_150.txt')
VOID_TEST_PREDICT_DEPTH_500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_predict_depth_500.txt')
VOID_TEST_PREDICT_DEPTH_1500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_predict_depth_1500.txt')


def run(model, dataloader, output_paths=None, verbose=False):
    '''
    Runs ScaffNet model
    if output paths are provided, then will save outputs to a predetermined list of paths

    Arg(s):
        model : ScaffNetModel
            ScaffNet model instance
        dataloader : torch.utils.data.DataLoader
            dataloader that outputs an image and a range map
        output_paths : list[str]
            list of paths to store output depth
        verbose : bool
            if set, then print progress to console
    Returns:
        list[numpy[float32]] : list of depth maps if output paths is None else no return value
    '''

    output_depths = []

    n_sample = len(dataloader)

    if output_paths is not None:
        assert len(output_paths) == n_sample

    for idx, sparse_depth in enumerate(dataloader):

        # Move inputs to device
        sparse_depth = sparse_depth.to(model.device)

        n_height, n_width = sparse_depth.shape[-2:]

        with torch.no_grad():
            # Forward through the network
            output_depth = model.forward(sparse_depth)

            if 'uncertainty' in model.decoder_type:
                output_depth = output_depth[:, 0:1, :, :]

            assert output_depth.shape[-2] == n_height
            assert output_depth.shape[-1] == n_width

        # Convert to numpy (if not converted already)
        output_depth = np.squeeze(output_depth.detach().cpu().numpy())

        if verbose:
            print('Processed {}/{} samples'.format(idx + 1, n_sample), end='\r')

        # Return output depths as a list if we do not store them
        if output_paths is None:
            output_depths.append(output_depth)
        else:
            data_utils.save_depth(output_depth, output_paths[idx])

    if output_paths is None:
        return output_depths


'''
Get commandline arguments
'''
parser = argparse.ArgumentParser()

# Input model
parser.add_argument('--restore_path',
    type=str, required=True, help='Path to restore checkpoint')
parser.add_argument('--paths_only',
    action='store_true', help='If set, the produce paths without generating dataset')

# Spatial pyramid pool settings
parser.add_argument('--max_pool_sizes_spatial_pyramid_pool',
    nargs='+', type=int, default=[13, 17, 19, 21, 25], help='List of pool sizes for spatial pyramid pooling')
parser.add_argument('--n_convolution_spatial_pyramid_pool',
    type=int, default=3, help='Number of convolutions to use for spatial pyramid pooling')
parser.add_argument('--n_filter_spatial_pyramid_pool',
    type=int, default=8, help='Number of filters to use for spatial pyramid pooling')

# Network settings
parser.add_argument('--encoder_type',
    nargs='+', type=str, default=['vggnet08', 'spatial_pyramid_pool', 'batch_norm'], help='Encoder type')
parser.add_argument('--n_filters_encoder',
    nargs='+', type=int, default=[16, 32, 64, 128, 256], help='Number of filters to each in each encoder block')
parser.add_argument('--decoder_type',
    nargs='+', type=str, default=['multi-scale', 'uncertainty', 'batch_norm'], help='Decoder type')
parser.add_argument('--n_filters_decoder',
    nargs='+', type=int, default=[256, 128, 128, 64, 32], help='Number of filters to each in each decoder block')
parser.add_argument('--min_predict_depth',
    type=float, default=0.1, help='Minimum value of depth prediction')
parser.add_argument('--max_predict_depth',
    type=float, default=10.0, help='Maximum value of depth prediction')

# Weight settings
parser.add_argument('--weight_initializer',
    type=str, default='xavier_normal', help='Weight initializers')
parser.add_argument('--activation_func',
    type=str, default='leaky_relu', help='Activation function')

# Hardware settings
parser.add_argument('--device',
    type=str, default='cuda', help='Device to use: gpu, cpu')

args = parser.parse_args()

'''
Assert arguments
'''
args.encoder_type = [
    encoder_type.lower() for encoder_type in args.encoder_type
]

assert len(args.n_filters_encoder) == 5

args.decoder_type = [
    decoder_type.lower() for decoder_type in args.decoder_type
]

assert len(args.n_filters_decoder) == 5

args.weight_initializer = args.weight_initializer.lower()

args.activation_func = args.activation_func.lower()

args.device = args.device.lower()

if args.device not in ['gpu', 'cpu', 'cuda']:
    args.device = 'cuda'

args.device = 'cuda' if args.device == 'gpu' else args.device


'''
Construct network
'''
model = ScaffNetModel(
    max_pool_sizes_spatial_pyramid_pool=args.max_pool_sizes_spatial_pyramid_pool,
    n_convolution_spatial_pyramid_pool=args.n_convolution_spatial_pyramid_pool,
    n_filter_spatial_pyramid_pool=args.n_filter_spatial_pyramid_pool,
    encoder_type=args.encoder_type,
    n_filters_encoder=args.n_filters_encoder,
    decoder_type=args.decoder_type,
    n_filters_decoder=args.n_filters_decoder,
    weight_initializer=args.weight_initializer,
    activation_func=args.activation_func,
    min_predict_depth=args.min_predict_depth,
    max_predict_depth=args.max_predict_depth,
    device=args.device)

# Restore model and set to evaluation mode
try:
    model.restore_model(args.restore_path)
except Exception:
    model.data_parallel()
    model.restore_model(args.restore_path)

model.eval()

'''
Set up input and output file paths
'''
input_output_filepaths = [
    [
        'training',
        '150',
        VOID_TRAIN_SPARSE_DEPTH_150_FILEPATH,
        VOID_TRAIN_PREDICT_DEPTH_150_FILEPATH
    ], [
        'training',
        '500',
        VOID_TRAIN_SPARSE_DEPTH_500_FILEPATH,
        VOID_TRAIN_PREDICT_DEPTH_500_FILEPATH
    ], [
        'training',
        '1500',
        VOID_TRAIN_SPARSE_DEPTH_1500_FILEPATH,
        VOID_TRAIN_PREDICT_DEPTH_1500_FILEPATH
    ], [
        'testing',
        '150',
        VOID_TEST_SPARSE_DEPTH_150_FILEPATH,
        VOID_TEST_PREDICT_DEPTH_150_FILEPATH
    ], [
        'testing',
        '500',
        VOID_TEST_SPARSE_DEPTH_500_FILEPATH,
        VOID_TEST_PREDICT_DEPTH_500_FILEPATH
    ], [
        'testing',
        '1500',
        VOID_TEST_SPARSE_DEPTH_1500_FILEPATH,
        VOID_TEST_PREDICT_DEPTH_1500_FILEPATH
    ]
]

for tag, density, input_filepath, output_filepath in input_output_filepaths:

    '''
    Construct output paths from input paths
    '''
    input_paths = data_utils.read_paths(input_filepath)

    n_sample = len(input_paths)

    output_paths = [
        input_path \
            .replace(VOID_DATA_DIRPATH, VOID_DATA_DERIVED_DIRPATH) \
            .replace('sparse_depth', 'predict_depth')
        for input_path in input_paths
    ]

    # Create directories first
    output_dirpaths = set([os.path.dirname(path) for path in output_paths])

    for dirpath in output_dirpaths:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    '''
    Generate predictions
    '''
    # Instantiate dataloader
    dataloader = torch.utils.data.DataLoader(
        datasets.ScaffNetInferenceDataset(depth_paths=input_paths),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    if not args.paths_only:
        print('Generating predicted depth for {} {} set {} density samples'.format(
            n_sample, tag, density))

        # Write to pseudo ground truth to disk
        run(model, dataloader, output_paths=output_paths, verbose=True)

    print('Storing predicted depth for {} {} set {} density sample file paths into: {}'.format(
        n_sample,
        tag,
        density,
        output_filepath))
    data_utils.write_paths(
        output_filepath,
        output_paths)
