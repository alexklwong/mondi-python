import argparse
from fusionnet_main import train


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--train_images_path',
    type=str, required=True, help='Path to list of training image paths')
parser.add_argument('--train_dense_depth_path',
    type=str, required=True, help='Path to list of training dense depth paths')
parser.add_argument('--train_sparse_depth_path',
    type=str, required=True, help='Path to list of training sparse depth paths')
parser.add_argument('--train_intrinsics_path',
    type=str, required=True, help='Path to list of training camera intrinsics paths')
parser.add_argument('--val_image_path',
    type=str, default='', help='Path to list of validation image paths')
parser.add_argument('--val_dense_depth_path',
    type=str, default='', help='Path to list of validation dense depth paths')
parser.add_argument('--val_sparse_depth_path',
    type=str, default='', help='Path to list of validation sparse depth paths')
parser.add_argument('--val_ground_truth_path',
    type=str, default='', help='Path to list of validation ground truth depth paths')

# Batch parameters
parser.add_argument('--n_batch',
    type=int, default=8, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, default=320, help='Height of of sample')
parser.add_argument('--n_width',
    type=int, default=768, help='Width of each sample')

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

# Training settings
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=[2.00e-4, 1.00e-4, 0.50e-4], help='Space delimited list of learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=[18, 24, 30], help='Space delimited list to change learning rate')
parser.add_argument('--augmentation_random_crop_type',
    nargs='+', type=str, default=['none'], help='Random crop types')

# Loss function settings
parser.add_argument('--w_color',
    type=float, default=0.20, help='Weight of color consistency loss')
parser.add_argument('--w_structure',
    type=float, default=0.80, help='Weight of structural consistency loss')
parser.add_argument('--w_sparse_depth',
    type=float, default=0.10, help='Weight of sparse depth consistency loss')
parser.add_argument('--w_smoothness',
    type=float, default=0.01, help='Weight of local smoothness loss')
parser.add_argument('--w_prior_depth',
    type=float, default=0.10, help='Weight of prior depth consistency loss')
parser.add_argument('--threshold_prior_depth',
    type=float, default=0.40, help='Threshold to cross before using prior depth loss')
parser.add_argument('--w_weight_decay_depth',
    type=float, default=0.00, help='Weight of weight decay regularization for depth')
parser.add_argument('--w_weight_decay_pose',
    type=float, default=0.00, help='Weight of weight decay regularization for pose')

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0.00, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=100.0, help='Maximum value of depth to evaluate')

# Checkpoint settings
parser.add_argument('--n_summary',
    type=int, default=5000, help='Number of iterations for logging summary')
parser.add_argument('--n_summary_display',
    type=int, default=4, help='Number of images to display when logging summary')
parser.add_argument('--n_checkpoint',
    type=int, default=5000, help='Number of iterations for each checkpoint')
parser.add_argument('--checkpoint_path',
    type=str, default=None, help='Path to save checkpoints')
parser.add_argument('--depth_model_restore_path',
    type=str, default=None, help='Path to restore depth model from checkpoint')
parser.add_argument('--pose_model_restore_path',
    type=str, default=None, help='Path to restore pose model from checkpoint')

# Hardware settings
parser.add_argument('--device',
    type=str, default='gpu', help='Device to use: gpu, cpu')
parser.add_argument('--n_thread',
    type=int, default=8, help='Number of threads for fetching')


args = parser.parse_args()

if __name__ == '__main__':

    assert len(args.learning_rates) == len(args.learning_schedule)

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

    train(train_images_path=args.train_images_path,
          train_dense_depth_path=args.train_dense_depth_path,
          train_sparse_depth_path=args.train_sparse_depth_path,
          train_intrinsics_path=args.train_intrinsics_path,
          val_image_path=args.val_image_path,
          val_dense_depth_path=args.val_dense_depth_path,
          val_sparse_depth_path=args.val_sparse_depth_path,
          val_ground_truth_path=args.val_ground_truth_path,
          # Batch settings
          n_batch=args.n_batch,
          n_height=args.n_height,
          n_width=args.n_width,
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
          # Training settings
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          augmentation_random_crop_type=args.augmentation_random_crop_type,
          # Loss function settings
          w_color=args.w_color,
          w_structure=args.w_structure,
          w_sparse_depth=args.w_sparse_depth,
          w_smoothness=args.w_smoothness,
          w_prior_depth=args.w_prior_depth,
          threshold_prior_depth=args.threshold_prior_depth,
          w_weight_decay_depth=args.w_weight_decay_depth,
          w_weight_decay_pose=args.w_weight_decay_pose,
          # Evaluation settings
          min_evaluate_depth=args.min_evaluate_depth,
          max_evaluate_depth=args.max_evaluate_depth,
          # Checkpoint settings
          n_summary=args.n_summary,
          n_summary_display=args.n_summary_display,
          n_checkpoint=args.n_checkpoint,
          checkpoint_path=args.checkpoint_path,
          depth_model_restore_path=args.depth_model_restore_path,
          pose_model_restore_path=args.pose_model_restore_path,
          # Hardware settings
          device=args.device,
          n_thread=args.n_thread)
