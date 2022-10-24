import argparse
from fusionnet_main import run


parser = argparse.ArgumentParser()

# Input filepaths
parser.add_argument('--image_path',
    type=str, default='', help='Path to list of image paths')
parser.add_argument('--dense_depth_path',
    type=str, default='', help='Path to list of dense depth paths')
parser.add_argument('--sparse_depth_path',
    type=str, default='', help='Path to list of sparse depth paths')
parser.add_argument('--ground_truth_path',
    type=str, default='', help='Path to list of ground truth depth paths')

# Input settings
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=[0, 1], help='Range of image intensities after normalization')
parser.add_argument('--outlier_removal_kernel_size',
    type=int, default=7, help='Kernel size to filter outlier sparse depth')
parser.add_argument('--outlier_removal_threshold',
    type=float, default=1.5, help='Difference threshold to consider a point an outlier')

# Network settings
parser.add_argument('--encoder_type',
    nargs='+', type=str, default='vggnet08', help='Encoder type')
parser.add_argument('--n_filters_encoder_image',
    nargs='+', type=int, default=[48, 96, 192, 384, 384], help='Space delimited list of filters to use in each block of image encoder')
parser.add_argument('--n_filters_encoder_depth',
    nargs='+', type=int, default=[16, 32, 64, 128, 128], help='Space delimited list of filters to use in each block of depth encoder')
parser.add_argument('--decoder_type',
    nargs='+', type=str, default='multi-scale', help='Decoder type')
parser.add_argument('--n_filters_decoder',
    nargs='+', type=int, default=[256, 128, 128, 64, 32], help='Space delimited list of filters to use in each block of decoder')
parser.add_argument('--scale_match_method',
    type=str, default='local_scale', help='Scale matching method')
parser.add_argument('--scale_match_kernel_size',
    type=int, default=5, help='Kernel size for local scale matching')
parser.add_argument('--min_predict_depth',
    type=float, default=1.5, help='Minimum value of predicted depth')
parser.add_argument('--max_predict_depth',
    type=float, default=100.0, help='Maximum value of predicted depth')
parser.add_argument('--min_multiplier_depth',
    type=float, default=0.25, help='Minimum value of depth multiplier')
parser.add_argument('--max_multiplier_depth',
    type=float, default=4.00, help='Maximum value of depth multiplier')
parser.add_argument('--min_residual_depth',
    type=float, default=-1000.0, help='Maximum value of depth residual')
parser.add_argument('--max_residual_depth',
    type=float, default=1000.0, help='Maximum value of depth residual')

# Weight settings
parser.add_argument('--weight_initializer',
    type=str, default='xavier_normal', help='Weight initializer')
parser.add_argument('--activation_func',
    type=str, default='leaky_relu', help='Activation function')

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0.00, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=100.0, help='Maximum value of depth to evaluate')

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

    args.encoder_type = [
        encoder_type.lower() for encoder_type in args.encoder_type
    ]

    assert len(args.n_filters_encoder_image) == 5
    assert len(args.n_filters_encoder_depth) == 5

    args.decoder_type = [
        decoder_type.lower() for decoder_type in args.decoder_type
    ]

    assert len(args.n_filters_decoder) == 5

    args.weight_initializer = args.weight_initializer.lower()

    args.activation_func = args.activation_func.lower()

    args.scale_match_method = args.scale_match_method.lower()

    args.device = args.device.lower()
    if args.device not in ['gpu', 'cpu', 'cuda']:
        args.device = 'cuda'

    args.device = 'cuda' if args.device == 'gpu' else args.device

    run(image_path=args.image_path,
        dense_depth_path=args.dense_depth_path,
        sparse_depth_path=args.sparse_depth_path,
        ground_truth_path=args.ground_truth_path,
        # Input settings
        normalized_image_range=args.normalized_image_range,
        outlier_removal_kernel_size=args.outlier_removal_kernel_size,
        outlier_removal_threshold=args.outlier_removal_threshold,
        # Depth network settings
        encoder_type=args.encoder_type,
        n_filters_encoder_image=args.n_filters_encoder_image,
        n_filters_encoder_depth=args.n_filters_encoder_depth,
        decoder_type=args.decoder_type,
        n_filters_decoder=args.n_filters_decoder,
        scale_match_method=args.scale_match_method,
        scale_match_kernel_size=args.scale_match_kernel_size,
        min_predict_depth=args.min_predict_depth,
        max_predict_depth=args.max_predict_depth,
        min_multiplier_depth=args.min_multiplier_depth,
        max_multiplier_depth=args.max_multiplier_depth,
        min_residual_depth=args.min_residual_depth,
        max_residual_depth=args.max_residual_depth,
        # Weight settings
        weight_initializer=args.weight_initializer,
        activation_func=args.activation_func,
        # Evaluation settings
        min_evaluate_depth=args.min_evaluate_depth,
        max_evaluate_depth=args.max_evaluate_depth,
        # Checkpoint settings
        checkpoint_path=args.checkpoint_path,
        restore_path=args.restore_path,
        # Output settings
        save_outputs=args.save_outputs,
        keep_input_filenames=args.keep_input_filenames,
        # Hardware settings
        device=args.device)
