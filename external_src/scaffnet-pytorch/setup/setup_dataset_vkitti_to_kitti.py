import os, sys, argparse
import numpy as np
import torch
sys.path.insert(0, 'src')
import data_utils, datasets
from scaffnet_model import ScaffNetModel
from log_utils import log
import global_constants as settings


'''
Data directories
'''
KITTI_DEPTH_COMPLETION_DIRPATH = os.path.join(
    'data', 'kitti_depth_completion')

KITTI_DEPTH_COMPLETION_OUTPUT_DIRPATH = os.path.join(
    'data', 'kitti_depth_completion_scaffnet')

TRAIN_OUTPUT_REF_DIRPATH = 'training'
VAL_OUTPUT_REF_DIRPATH = 'validation'
TEST_OUTPUT_REF_DIRPATH = 'testing'


'''
Input file paths
'''
TRAIN_SPARSE_DEPTH_INPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_sparse_depth.txt')
TRAIN_VALIDITY_MAP_INPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_validity_map.txt')

TRAIN_SPARSE_DEPTH_CLEAN_INPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_sparse_depth-clean.txt')
TRAIN_VALIDITY_MAP_CLEAN_INPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_validity_map-clean.txt')

UNUSED_SPARSE_DEPTH_INPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_unused_sparse_depth.txt')
UNUSED_VALIDITY_MAP_INPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_unused_validity_map.txt')

VAL_SPARSE_DEPTH_INPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH, 'kitti_val_sparse_depth.txt')
VAL_VALIDITY_MAP_INPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH, 'kitti_val_validity_map.txt')

TEST_SPARSE_DEPTH_INPUT_FILEPATH = os.path.join(
    TEST_OUTPUT_REF_DIRPATH, 'kitti_test_sparse_depth.txt')
TEST_VALIDITY_MAP_INPUT_FILEPATH = os.path.join(
    TEST_OUTPUT_REF_DIRPATH, 'kitti_test_validity_map.txt')


'''
Output file paths
'''
TRAIN_PREDICT_DEPTH_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_predict_depth.txt')

TRAIN_PREDICT_DEPTH_CLEAN_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_predict_depth-clean.txt')

UNUSED_PREDICT_DEPTH_OUTPUT_FILEPATH = os.path.join(
    TRAIN_OUTPUT_REF_DIRPATH, 'kitti_unused_predict_depth.txt')

VAL_PREDICT_DEPTH_OUTPUT_FILEPATH = os.path.join(
    VAL_OUTPUT_REF_DIRPATH, 'kitti_val_predict_depth.txt')

TEST_PREDICT_DEPTH_OUTPUT_FILEPATH = os.path.join(
    TEST_OUTPUT_REF_DIRPATH, 'kitti_test_predict_depth.txt')

'''
Get commandline arguments
'''
parser = argparse.ArgumentParser()

# Input model
parser.add_argument('--restore_path',
    type=str, required=True, help='Path to restore checkpoint')
# Network settings
parser.add_argument('--encoder_type',
    nargs='+', type=str, default=settings.ENCODER_TYPE_SCAFFNET, help='Encoder type: %s' % settings.ENCODER_TYPE_AVAILABLE_SCAFFNET)
parser.add_argument('--n_filters_encoder',
    nargs='+', type=int, default=settings.N_FILTERS_ENCODER_SCAFFNET, help='Number of filters to each in each encoder block')
parser.add_argument('--decoder_type',
    nargs='+', type=str, default=settings.DECODER_TYPE_SCAFFNET, help='Decoder type: %s' % settings.DECODER_TYPE_AVAILABLE_SCAFFNET)
parser.add_argument('--n_filters_decoder',
    nargs='+', type=int, default=settings.N_FILTERS_DECODER_SCAFFNET, help='Number of filters to each in each decoder block')
parser.add_argument('--deconv_type',
    type=str, default=settings.DECONV_TYPE_SCAFFNET, help='Deconvolution type: %s' % settings.DECONV_TYPE_AVAILABLE_SCAFFNET)
parser.add_argument('--spatial_pyramid_pool_sizes',
    nargs='+', type=int, default=settings.SPATIAL_PYRAMID_POOL_SIZES, help='List of pool sizes for spatial pyramid pooling')
parser.add_argument('--n_conv_spatial_pyramid_pool',
    type=int, default=settings.N_CONV_SPATIAL_PYRAMID_POOL, help='Number of convolutions to use for spatial pyramid pooling')
parser.add_argument('--n_filter_spatial_pyramid_pool',
    type=int, default=settings.N_FILTER_SPATIAL_PYRAMID_POOL, help='Number of filters to use for spatial pyramid pooling')
parser.add_argument('--weight_initializer',
    type=str, default=settings.WEIGHT_INITIALIZER, help='Weight initializers: %s' % settings.WEIGHT_INITIALIZER_AVAILABLE)
parser.add_argument('--activation_func',
    type=str, default=settings.ACTIVATION_FUNC, help='Activation function: %s' % settings.ACTIVATION_FUNC_AVAILABLE)
parser.add_argument('--output_func',
    type=str, default=settings.OUTPUT_FUNC_SCAFFNET, help='Output function: %s' % settings.OUTPUT_FUNC_AVAILABLE_SCAFFNET)
parser.add_argument('--use_batch_norm',
    action='store_true', help='If set, then use batch normalization')
# Depth range settings
parser.add_argument('--min_predict_depth',
    type=float, default=settings.MIN_PREDICT_DEPTH, help='Minimum value of depth prediction')
parser.add_argument('--max_predict_depth',
    type=float, default=settings.MAX_PREDICT_DEPTH, help='Maximum value of depth prediction')
# Hardware settings
parser.add_argument('--device',
    type=str, default=settings.DEVICE, help='Device to use: gpu, cpu')


args = parser.parse_args()


'''
Assert and sanitize input arguments
'''
assert(os.path.exists(args.restore_path))

args.encoder_type = [
    encoder_type.lower() for encoder_type in args.encoder_type
]
for encoder_type in args.encoder_type:
    assert(encoder_type in settings.ENCODER_TYPE_AVAILABLE_SCAFFNET)

assert(len(args.n_filters_encoder) == 5)

args.decoder_type = [
    decoder_type.lower() for decoder_type in args.decoder_type
]
for decoder_type in args.decoder_type:
    assert(decoder_type in settings.DECODER_TYPE_AVAILABLE_SCAFFNET)

assert(len(args.n_filters_decoder) == 5)

args.deconv_type = args.deconv_type.lower()
assert(args.deconv_type in settings.DECONV_TYPE_AVAILABLE_SCAFFNET)

args.weight_initializer = args.weight_initializer.lower()
assert(args.weight_initializer in settings.WEIGHT_INITIALIZER_AVAILABLE)

args.activation_func = args.activation_func.lower()
assert(args.activation_func in settings.ACTIVATION_FUNC_AVAILABLE)

args.output_func = args.output_func.lower()
assert(args.output_func in settings.OUTPUT_FUNC_AVAILABLE_SCAFFNET)

args.device = args.device.lower()
if args.device not in [settings.GPU, settings.CPU, settings.CUDA]:
    args.device = settings.CUDA

args.device = settings.CUDA if args.device == settings.GPU else args.device

if args.device == settings.CUDA or args.device == settings.GPU:
    device = torch.device(settings.CUDA)
else:
    device = torch.device(settings.CPU)


'''
Construct output paths from input paths
'''
train_sparse_depth_paths = data_utils.read_paths(TRAIN_SPARSE_DEPTH_INPUT_FILEPATH)
train_sparse_depth_clean_paths = data_utils.read_paths(TRAIN_SPARSE_DEPTH_CLEAN_INPUT_FILEPATH)
unused_sparse_depth_paths = data_utils.read_paths(UNUSED_SPARSE_DEPTH_INPUT_FILEPATH)
val_sparse_depth_paths = data_utils.read_paths(VAL_SPARSE_DEPTH_INPUT_FILEPATH)
test_sparse_depth_paths = data_utils.read_paths(TEST_SPARSE_DEPTH_INPUT_FILEPATH)

input_paths = [
    train_sparse_depth_paths,
    train_sparse_depth_clean_paths,
    unused_sparse_depth_paths,
    val_sparse_depth_paths,
    test_sparse_depth_paths
]

train_predict_depth_paths = []
train_predict_depth_clean_paths = []
unused_predict_depth_paths = []
val_predict_depth_paths = []
test_predict_depth_paths = []

output_paths = [
    train_predict_depth_paths,
    train_predict_depth_clean_paths,
    unused_predict_depth_paths,
    val_predict_depth_paths,
    test_predict_depth_paths
]

# Create output paths for predicted dense depth
for sparse_depth_paths, predict_depth_paths in zip(input_paths, output_paths):
    # Iterate through each sparse depth path and modify it to output path
    for path in sparse_depth_paths:
        predict_path = path \
            .replace(KITTI_DEPTH_COMPLETION_DIRPATH, KITTI_DEPTH_COMPLETION_OUTPUT_DIRPATH) \
            .replace('sparse_depth', 'predict_depth')
        predict_depth_paths.append(predict_path)

print('Storing training predicted depth file paths into: %s' %
    TRAIN_PREDICT_DEPTH_OUTPUT_FILEPATH)
data_utils.write_paths(
    TRAIN_PREDICT_DEPTH_OUTPUT_FILEPATH, train_predict_depth_paths)

print('Storing clean training predicted depth file paths into: %s' %
    TRAIN_PREDICT_DEPTH_CLEAN_OUTPUT_FILEPATH)
data_utils.write_paths(
    TRAIN_PREDICT_DEPTH_CLEAN_OUTPUT_FILEPATH, train_predict_depth_clean_paths)

print('Storing unused predicted depth file paths into: %s' %
    UNUSED_PREDICT_DEPTH_OUTPUT_FILEPATH)
data_utils.write_paths(
    UNUSED_PREDICT_DEPTH_OUTPUT_FILEPATH, unused_predict_depth_paths)

print('Storing validation predicted depth file paths into: %s' %
    VAL_PREDICT_DEPTH_OUTPUT_FILEPATH)
data_utils.write_paths(
    VAL_PREDICT_DEPTH_OUTPUT_FILEPATH, val_predict_depth_paths)

print('Storing testing predicted depth file paths into: %s' %
    TEST_PREDICT_DEPTH_OUTPUT_FILEPATH)
data_utils.write_paths(
    TEST_PREDICT_DEPTH_OUTPUT_FILEPATH, test_predict_depth_paths)


# Create the lists without clean paths since they are subset of full training
input_paths = [
    train_sparse_depth_paths,
    unused_sparse_depth_paths,
    val_sparse_depth_paths,
    test_sparse_depth_paths
]

output_paths = [
    train_predict_depth_paths,
    unused_predict_depth_paths,
    val_predict_depth_paths,
    test_predict_depth_paths
]


'''
Construct dataloader and network
'''
with torch.no_grad():
    model = ScaffNetModel(
        encoder_type=args.encoder_type,
        n_filters_encoder=args.n_filters_encoder,
        decoder_type=args.decoder_type,
        n_filters_decoder=args.n_filters_decoder,
        deconv_type=args.deconv_type,
        spatial_pyramid_pool_sizes=args.spatial_pyramid_pool_sizes,
        n_conv_spatial_pyramid_pool=args.n_conv_spatial_pyramid_pool,
        n_filter_spatial_pyramid_pool=args.n_filter_spatial_pyramid_pool,
        weight_initializer=args.weight_initializer,
        activation_func=args.activation_func,
        output_func=args.output_func,
        use_batch_norm=args.use_batch_norm,
        min_predict_depth=args.min_predict_depth,
        max_predict_depth=args.max_predict_depth,
        device=device)

    # Restore model and set to evaluation mode
    model.restore_model(args.restore_path)
    model.eval()

    model_parameters = model.parameters()
    n_param = sum(p.numel() for p in model_parameters)

    '''
    Log model arguments to console
    '''
    log('Network settings:')
    log('encoder_type=%s' %
        (args.encoder_type))
    log('n_filters_encoder=%s' %
        (args.n_filters_encoder))

    log('decoder_type=%s' %
        (args.decoder_type))
    log('n_filters_decoder=%s  deconv_type=%s  output_func=%s' %
        (args.n_filters_decoder, args.deconv_type, args.output_func))

    log('Spatial pyramid pooling settings:')
    log('pool_sizes=%s  n_convolution=%s  n_filter=%s' %
        (args.spatial_pyramid_pool_sizes, args.n_conv_spatial_pyramid_pool, args.n_filter_spatial_pyramid_pool))

    log('Weight settings:')
    log('n_param=%d' %
        (n_param))
    log('weight_initializer=%s  activation_func=%s  use_batch_norm=%s' %
        (args.weight_initializer, args.activation_func, args.use_batch_norm))

    log('Depth range settings:')
    log('min_predict_depth=%.2f  max_predict_depth=%.2f' %
        (args.min_predict_depth, args.max_predict_depth))

    log('Checkpoint settings:')
    log('restore_path=%s' % args.restore_path)

    # Run model through paths
    for paths_idx, (sparse_depth_paths, predict_depth_paths) in enumerate(zip(input_paths, output_paths)):

        dataloader = torch.utils.data.DataLoader(
            datasets.DepthDataset(depth_paths=sparse_depth_paths),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)

        for sample_idx, sparse_depth in enumerate(dataloader):

            sys.stdout.write(
                'Processing {}/{} examples in path list {}/{} \r'.format(
                    sample_idx + 1, len(dataloader), paths_idx + 1, len(output_paths)))
            sys.stdout.flush()

            if device.type == settings.CUDA:
                sparse_depth = sparse_depth.cuda()

            # Forward through network
            output_depth = model.forward(sparse_depth)

            # Convert to numpy
            if device.type == settings.CUDA:
                output_depth = np.squeeze(output_depth.cpu().numpy())
            else:
                output_depth = np.squeeze(output_depth.numpy())

            # Save depth to file
            if not os.path.exists(os.path.dirname(predict_depth_paths[sample_idx])):
                os.makedirs(os.path.dirname(predict_depth_paths[sample_idx]))

            data_utils.save_depth(output_depth, predict_depth_paths[sample_idx])
