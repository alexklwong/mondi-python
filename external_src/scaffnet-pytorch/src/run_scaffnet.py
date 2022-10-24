import argparse
from scaffnet_main import run


parser = argparse.ArgumentParser()

# Input filepaths
parser.add_argument('--sparse_depth_path',
    type=str, required=True, help='Path to list of sparse depth paths')
parser.add_argument('--ground_truth_path',
    type=str, required=None, help='Path to list ofgroundtruth depth paths')

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

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0.2, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=10.0, help='Maximum value of depth to evaluate')

# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--restore_path',
    type=str, required=True, help='Path to restore model from checkpoint')
parser.add_argument('--save_outputs',
    action='store_true', help='If set then store inputs and outputs into output path')
parser.add_argument('--keep_input_filenames',
    action='store_true', help='If set then keep original input filenames')

# Hardware settings
parser.add_argument('--device',
    type=str, default='cuda', help='Device to use: gpu, cpu')


args = parser.parse_args()

if __name__ == '__main__':

    # Assert arguments
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

    run(sparse_depth_path=args.sparse_depth_path,
        ground_truth_path=args.ground_truth_path,
        # Spatial pyramid pool settings
        max_pool_sizes_spatial_pyramid_pool=args.max_pool_sizes_spatial_pyramid_pool,
        n_convolution_spatial_pyramid_pool=args.n_convolution_spatial_pyramid_pool,
        n_filter_spatial_pyramid_pool=args.n_filter_spatial_pyramid_pool,
        # Network settings
        encoder_type=args.encoder_type,
        n_filters_encoder=args.n_filters_encoder,
        decoder_type=args.decoder_type,
        n_filters_decoder=args.n_filters_decoder,
        min_predict_depth=args.min_predict_depth,
        max_predict_depth=args.max_predict_depth,
        # Weight settings
        weight_initializer=args.weight_initializer,
        activation_func=args.activation_func,
        # Evaluation settings
        min_evaluate_depth=args.min_evaluate_depth,
        max_evaluate_depth=args.max_evaluate_depth,
        # Checkpoint settings
        checkpoint_path=args.checkpoint_path,
        restore_path=args.restore_path,
        save_outputs=args.save_outputs,
        keep_input_filenames=args.keep_input_filenames,
        # Hardware settings
        device=args.device)
