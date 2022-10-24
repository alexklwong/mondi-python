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

import argparse, os
import global_constants as settings
from mondi import run


parser = argparse.ArgumentParser()


parser.add_argument('--image_path',
    type=str, required=True, help='Path to list of image paths')
parser.add_argument('--sparse_depth_path',
    type=str, required=True, help='Path to list of sparse depth paths')
parser.add_argument('--intrinsics_path',
    type=str, required=True, help='Path to list of camera intrinsics paths')
parser.add_argument('--ground_truth_path',
    type=str, default=None, help='Path to list of ground truth depth paths')

# Input settings
parser.add_argument('--load_image_triplets',
    action='store_true', help='If set then load image triplets')
parser.add_argument('--input_types',
    nargs='+', type=str, default=settings.INPUT_TYPES, help='Inputs to network')
parser.add_argument('--input_channels_image',
    type=int, default=settings.INPUT_CHANNELS_IMAGE, help='Number of input image channels')
parser.add_argument('--input_channels_depth',
    type=int, default=settings.INPUT_CHANNELS_DEPTH, help='Number of input depth channels')
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=settings.NORMALIZED_IMAGE_RANGE, help='Range of image intensities after normalization')
parser.add_argument('--outlier_removal_kernel_size',
    type=int, default=settings.OUTLIER_REMOVAL_KERNEL_SIZE, help='Kernel size to filter outlier sparse depth')
parser.add_argument('--outlier_removal_threshold',
    type=float, default=settings.OUTLIER_REMOVAL_THRESHOLD, help='Difference threshold to consider a point an outlier')

# Sparse to dense pool settings
parser.add_argument('--min_pool_sizes_sparse_to_dense_pool',
    nargs='+', type=int, default=settings.MIN_POOL_SIZES_SPARSE_TO_DENSE_POOL, help='Space delimited list of min pool sizes for sparse to dense pooling')
parser.add_argument('--max_pool_sizes_sparse_to_dense_pool',
    nargs='+', type=int, default=settings.MAX_POOL_SIZES_SPARSE_TO_DENSE_POOL, help='Space delimited list of max pool sizes for sparse to dense pooling')
parser.add_argument('--n_convolution_sparse_to_dense_pool',
    type=int, default=settings.N_CONVOLUTION_SPARSE_TO_DENSE_POOL, help='Number of convolutions for sparse to dense pooling')
parser.add_argument('--n_filter_sparse_to_dense_pool',
    type=int, default=settings.N_FILTER_SPARSE_TO_DENSE_POOL, help='Number of filters for sparse to dense pooling')

# Depth network settings
parser.add_argument('--encoder_type',
    nargs='+', type=str, default=settings.ENCODER_TYPE, help='Encoder type: resnet18')
parser.add_argument('--n_filters_encoder_image',
    nargs='+', type=int, default=settings.N_FILTERS_ENCODER_IMAGE, help='Space delimited list of filters to use in each block of image encoder')
parser.add_argument('--n_filters_encoder_depth',
    nargs='+', type=int, default=settings.N_FILTERS_ENCODER_DEPTH, help='Space delimited list of filters to use in each block of depth encoder')
parser.add_argument('--n_convolutions_encoder',
    nargs='+', type=int, default=settings.N_CONVOLUTIONS_ENCODER, help='Space delimited list of convolutions to use for encoder')
parser.add_argument('--resolutions_backprojection',
    nargs='+', type=int, default=settings.RESOLUTIONS_BACKPROJECTION, help='Space delimited list of resolutions to use calibrated backprojection')
parser.add_argument('--resolutions_depthwise_separable_encoder',
    nargs='+', type=int, default=settings.RESOLUTIONS_DEPTHWISE_SEPARABLE_ENCODER, help='Space delimited list of resolutions to use depthwise separable')
parser.add_argument('--decoder_type',
    nargs='+', type=str, default=settings.DECODER_TYPE, help='Decoder type: multiscale')
parser.add_argument('--n_filters_decoder',
    nargs='+', type=int, default=settings.N_FILTERS_DECODER, help='Space delimited list of filters to use in each block of depth decoder')
parser.add_argument('--n_resolution_decoder',
    type=int, default=settings.N_RESOLUTION_DECODER, help='Number of resolutions for multiscale outputs')
parser.add_argument('--resolutions_depthwise_separable_decoder',
    nargs='+', type=int, default=settings.RESOLUTIONS_DEPTHWISE_SEPARABLE_DECODER, help='Space delimited list of resolutions to use depthwise separable')
parser.add_argument('--min_predict_depth',
    type=float, default=settings.MIN_PREDICT_DEPTH, help='Minimum value of predicted depth')
parser.add_argument('--max_predict_depth',
    type=float, default=settings.MAX_PREDICT_DEPTH, help='Maximum value of predicted depth')

# Weight settings
parser.add_argument('--weight_initializer',
    type=str, default=settings.WEIGHT_INITIALIZER, help='Weight initialization type: kaiming_uniform, kaiming_normal, xavier_uniform, xavier_normal')
parser.add_argument('--activation_func',
    type=str, default=settings.ACTIVATION_FUNC, help='Activation function after each layer: relu, leaky_relu, elu, sigmoid')

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=settings.MIN_EVALUATE_DEPTH, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=settings.MAX_EVALUATE_DEPTH, help='Maximum value of depth to evaluate')

# Checkpoint settings
parser.add_argument('--output_path',
    type=str, default=settings.CHECKPOINT_PATH, help='Path to save checkpoints')
parser.add_argument('--depth_model_restore_path',
    type=str, default=settings.RESTORE_PATH, help='Path to restore depth model from checkpoint')
parser.add_argument('--save_outputs',
    action='store_true', help='If set then store inputs and outputs into output path')
parser.add_argument('--keep_input_filenames',
    action='store_true', help='If set then keep original input filenames')

# Hardware settings
parser.add_argument('--device',
    type=str, default=settings.DEVICE, help='Device to use: cuda, gpu, cpu')


args = parser.parse_args()

if __name__ == '__main__':

    '''
    Assert inputs
    '''
    # Weight settings
    args.weight_initializer = args.weight_initializer.lower()

    args.activation_func = args.activation_func.lower()

    # Checkpoint settings
    assert args.depth_model_restore_path is not None
    assert os.path.exists(args.depth_model_restore_path)

    # Hardware settings
    args.device = args.device.lower()
    if args.device not in [settings.GPU, settings.CPU, settings.CUDA]:
        args.device = settings.CUDA

    args.device = settings.CUDA if args.device == settings.GPU else args.device

    run(image_path=args.image_path,
        sparse_depth_path=args.sparse_depth_path,
        intrinsics_path=args.intrinsics_path,
        ground_truth_path=args.ground_truth_path,
        # Input settings
        load_image_triplets=args.load_image_triplets,
        input_types=args.input_types,
        input_channels_image=args.input_channels_image,
        input_channels_depth=args.input_channels_depth,
        normalized_image_range=args.normalized_image_range,
        outlier_removal_kernel_size=args.outlier_removal_kernel_size,
        outlier_removal_threshold=args.outlier_removal_threshold,
        # Sparse to dense pool settings
        min_pool_sizes_sparse_to_dense_pool=args.min_pool_sizes_sparse_to_dense_pool,
        max_pool_sizes_sparse_to_dense_pool=args.max_pool_sizes_sparse_to_dense_pool,
        n_convolution_sparse_to_dense_pool=args.n_convolution_sparse_to_dense_pool,
        n_filter_sparse_to_dense_pool=args.n_filter_sparse_to_dense_pool,
        # Depth network settings
        encoder_type=args.encoder_type,
        n_filters_encoder_image=args.n_filters_encoder_image,
        n_filters_encoder_depth=args.n_filters_encoder_depth,
        n_convolutions_encoder=args.n_convolutions_encoder,
        resolutions_backprojection=args.resolutions_backprojection,
        resolutions_depthwise_separable_encoder=args.resolutions_depthwise_separable_encoder,
        decoder_type=args.decoder_type,
        n_filters_decoder=args.n_filters_decoder,
        n_resolution_decoder=args.n_resolution_decoder,
        resolutions_depthwise_separable_decoder=args.resolutions_depthwise_separable_decoder,
        min_predict_depth=args.min_predict_depth,
        max_predict_depth=args.max_predict_depth,
        # Weight settings
        weight_initializer=args.weight_initializer,
        activation_func=args.activation_func,
        # Evaluation settings
        min_evaluate_depth=args.min_evaluate_depth,
        max_evaluate_depth=args.max_evaluate_depth,
        # Checkpoint settings
        checkpoint_path=args.output_path,
        depth_model_restore_path=args.depth_model_restore_path,
        # Output settings
        save_outputs=args.save_outputs,
        keep_input_filenames=args.keep_input_filenames,
        # Hardware settings
        device=args.device)
