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

import argparse
import torch
import global_constants as settings
from mondi import train


parser = argparse.ArgumentParser()


# Training input filepaths
parser.add_argument('--train_image0_path',
    type=str, required=True, help='Path to list of training left camera image paths')
parser.add_argument('--train_image1_path',
    type=str, default=None, help='Path to list of training right camera image paths')
parser.add_argument('--train_sparse_depth0_path',
    type=str, required=True, help='Path to list of training left camera sparse depth paths')
parser.add_argument('--train_sparse_depth1_path',
    type=str, default=None, help='Path to list of training right camera sparse depth paths')
parser.add_argument('--train_ground_truth0_path',
    type=str, required=True, help='Path to list of training left camera ground truth paths')
parser.add_argument('--train_ground_truth1_path',
    type=str, default=None, help='Path to list of training right camera ground truth paths')
parser.add_argument('--train_teacher_output0_paths',
    nargs='+', type=str, required=True, help='Path to list of training left camera teacher output paths')
parser.add_argument('--train_teacher_output1_paths',
    nargs='+', type=str, default=None, help='Path to list of training right camera teacher output paths')
parser.add_argument('--train_intrinsics0_path',
    type=str, required=True, help='Path to list of training left camera intrinsics paths')
parser.add_argument('--train_intrinsics1_path',
    type=str, default=None, help='Path to list of training right camera intrinsics paths')
parser.add_argument('--train_focal_length_baseline0_path',
    type=str, default=None, help='Path to list of training left camera focal length baseline paths')
parser.add_argument('--train_focal_length_baseline1_path',
    type=str, default=None, help='Path to list of training right camera focal length baseline paths')

# Validation input filepaths
parser.add_argument('--val_image_path',
    type=str, default=None, help='Path to list of validation image paths')
parser.add_argument('--val_sparse_depth_path',
    type=str, default=None, help='Path to list of validation sparse depth paths')
parser.add_argument('--val_intrinsics_path',
    type=str, default=None, help='Path to list of validation camera intrinsics paths')
parser.add_argument('--val_ground_truth_path',
    type=str, default=None, help='Path to list of validation ground truth depth paths')

# Batch parameters
parser.add_argument('--n_batch',
    type=int, default=settings.N_BATCH, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, default=settings.N_HEIGHT, help='Height of of sample')
parser.add_argument('--n_width',
    type=int, default=settings.N_WIDTH, help='Width of each sample')

# Input settings
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

# Training settings
parser.add_argument('--supervision_types',
    nargs='+', type=str, default=settings.SUPERVISION_TYPES, help='Space delimited list: monocular, stereo')
parser.add_argument('--learning_rates_depth',
    nargs='+', type=float, default=settings.LEARNING_RATES, help='Space delimited list of learning rates')
parser.add_argument('--learning_schedule_depth',
    nargs='+', type=int, default=settings.LEARNING_SCHEDULE, help='Space delimited list to change learning rate')
parser.add_argument('--learning_rates_pose',
    nargs='+', type=float, default=settings.LEARNING_RATES, help='Space delimited list of learning rates')
parser.add_argument('--learning_schedule_pose',
    nargs='+', type=int, default=settings.LEARNING_SCHEDULE, help='Space delimited list to change learning rate')

# Augmentation settings
parser.add_argument('--augmentation_probabilities',
    nargs='+', type=float, default=settings.AUGMENTATION_PROBABILITIES, help='Probabilities to use data augmentation. Note: there is small chance that no augmentation take place even when we enter augmentation pipeline.')
parser.add_argument('--augmentation_schedule',
    nargs='+', type=int, default=settings.AUGMENTATION_SCHEDULE, help='If not -1, then space delimited list to change augmentation probability')
parser.add_argument('--augmentation_random_swap',
    action='store_true', help='If set, peform random swapping between L and R images')
parser.add_argument('--augmentation_random_crop_type',
    nargs='+', type=str, default=settings.AUGMENTATION_RANDOM_CROP_TYPE, help='Random crop type for data augmentation: horizontal, vertical, anchored, bottom')
parser.add_argument('--augmentation_random_crop_to_shape',
    nargs='+', type=int, default=settings.AUGMENTATION_RANDOM_CROP_TO_SHAPE, help='Shape after cropping')
parser.add_argument('--augmentation_random_remove_points',
    nargs='+', type=float, default=settings.AUGMENTATION_RANDOM_REMOVE_POINTS, help='If set, randomly remove points from sparse depth')
parser.add_argument('--augmentation_random_brightness',
    nargs='+', type=float, default=settings.AUGMENTATION_RANDOM_BRIGHTNESS, help='If does not contain -1, apply random brightness')
parser.add_argument('--augmentation_random_contrast',
    nargs='+', type=float, default=settings.AUGMENTATION_RANDOM_CONTRAST, help='If does not contain -1, apply random contrast')
parser.add_argument('--augmentation_random_saturation',
    nargs='+', type=float, default=settings.AUGMENTATION_RANDOM_SATURATION, help='If does not contain -1, apply random saturation')

# Loss function settings
parser.add_argument('--w_stereo',
    type=float, default=settings.W_STEREO, help='Weight of stereo reconstruction loss terms')
parser.add_argument('--w_monocular',
    type=float, default=settings.W_MONOCULAR, help='Weight of monocular reconstruction loss terms')
parser.add_argument('--w_color',
    type=float, default=settings.W_COLOR, help='Weight of color consistency loss')
parser.add_argument('--w_structure',
    type=float, default=settings.W_STRUCTURE, help='Weight of structural consistency loss')
parser.add_argument('--w_sparse_depth',
    type=float, default=settings.W_SPARSE_DEPTH, help='Weight of sparse depth consistency loss')
parser.add_argument('--w_ensemble_depth',
    type=float, default=settings.W_ENSEMBLE_DEPTH, help='Weight of ensemble depth consistency loss')
parser.add_argument('--w_ensemble_temperature',
    type=float, default=settings.W_ENSEMBLE_TEMPERATURE, help='Temperature of ensemble adaptive weighting')
parser.add_argument('--w_sparse_select_ensemble',
    type=float, default=settings.W_SPARSE_SELECT_ENSEMBLE, help='Default weight of sparse depth consistency in teacher selection process')
parser.add_argument('--sparse_select_dilate_kernel_size',
    type=int, default=settings.SPARSE_SELECT_DILATE_KERNEL_SIZE, help='Dilation kernel_size for sparse depth consistency in teacher selection process')
parser.add_argument('--w_smoothness',
    type=float, default=settings.W_SMOOTHNESS, help='Weight of local smoothness loss')
parser.add_argument('--w_weight_decay_depth',
    type=float, default=settings.W_WEIGHT_DECAY_DEPTH, help='Weight of weight decay regularization for depth')
parser.add_argument('--w_weight_decay_pose',
    type=float, default=settings.W_WEIGHT_DECAY_POSE, help='Weight of weight decay regularization for depth')
parser.add_argument('--loss_func_ensemble_depth',
    type=str, default=settings.LOSS_FUNC_ENSEMBLE_DEPTH, help='Loss function to use for teacher output - l1, l2, smoothl1')
parser.add_argument('--epoch_pose_for_ensemble',
    type=int, default=settings.EPOCH_POSE_FOR_ENSEMBLE, help='Epoch at which to use posenet output for selecting from ensemble')
parser.add_argument('--use_supervised_adaptive_weights',
    action='store_true', help='If set, use photometric loss as weights for supervised loss')
parser.add_argument('--ensemble_method',
    type=str, default=settings.ENSEMBLE_METHOD, help='median|mean|random|mondi (default)')

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=settings.MIN_EVALUATE_DEPTH, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=settings.MAX_EVALUATE_DEPTH, help='Maximum value of depth to evaluate')

# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, default=settings.CHECKPOINT_PATH, help='Path to save checkpoints')
parser.add_argument('--n_checkpoint',
    type=int, default=settings.N_CHECKPOINT, help='Number of iterations for each checkpoint')
parser.add_argument('--n_summary',
    type=int, default=settings.N_SUMMARY, help='Number of iterations before logging summary')
parser.add_argument('--n_summary_display',
    type=int, default=settings.N_SUMMARY_DISPLAY, help='Number of samples to include in visual display summary')
parser.add_argument('--validation_start_step',
    type=int, default=settings.VALIDATION_START_STEP, help='Number of steps before starting validation')
parser.add_argument('--depth_model_restore_path',
    type=str, default=None, help='Path to restore depth model from checkpoint')
parser.add_argument('--pose_model_restore_path',
    type=str, default=None, help='Path to restore pose model from checkpoint')

# Hardware settings
parser.add_argument('--device',
    type=str, default=settings.DEVICE, help='Device to use: cuda, gpu, cpu')
parser.add_argument('--n_thread',
    type=int, default=settings.N_THREAD, help='Number of threads for fetching')


args = parser.parse_args()

if __name__ == '__main__':

    # Weight settings
    args.weight_initializer = args.weight_initializer.lower()

    args.activation_func = args.activation_func.lower()

    # Training settings
    args.supervision_types = [
        supervision_type.lower() for supervision_type in args.supervision_types
    ]

    assert len(args.learning_rates_depth) == len(args.learning_schedule_depth)
    assert len(args.learning_rates_pose) == len(args.learning_schedule_pose)

    args.augmentation_random_crop_type = [
        crop_type.lower() for crop_type in args.augmentation_random_crop_type
    ]

    # Hardware settings
    args.device = args.device.lower()
    if args.device not in settings.DEVICE_AVAILABLE:
        args.device = settings.CUDA if torch.cuda.is_available() else settings.CPU

    args.device = settings.CUDA if args.device == settings.GPU else args.device

    train(train_image0_path=args.train_image0_path,
          train_image1_path=args.train_image1_path,
          train_sparse_depth0_path=args.train_sparse_depth0_path,
          train_sparse_depth1_path=args.train_sparse_depth1_path,
          train_ground_truth0_path=args.train_ground_truth0_path,
          train_ground_truth1_path=args.train_ground_truth1_path,
          train_teacher_output0_paths=args.train_teacher_output0_paths,
          train_teacher_output1_paths=args.train_teacher_output1_paths,
          train_intrinsics0_path=args.train_intrinsics0_path,
          train_intrinsics1_path=args.train_intrinsics1_path,
          train_focal_length_baseline0_path=args.train_focal_length_baseline0_path,
          train_focal_length_baseline1_path=args.train_focal_length_baseline1_path,
          val_image_path=args.val_image_path,
          val_sparse_depth_path=args.val_sparse_depth_path,
          val_intrinsics_path=args.val_intrinsics_path,
          val_ground_truth_path=args.val_ground_truth_path,
          # Batch settings
          n_batch=args.n_batch,
          n_height=args.n_height,
          n_width=args.n_width,
          # Input settings
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
          # Training settings
          supervision_types=args.supervision_types,
          learning_rates_depth=args.learning_rates_depth,
          learning_schedule_depth=args.learning_schedule_depth,
          learning_rates_pose=args.learning_rates_pose,
          learning_schedule_pose=args.learning_schedule_pose,
          # Augmentation settings
          augmentation_probabilities=args.augmentation_probabilities,
          augmentation_schedule=args.augmentation_schedule,
          augmentation_random_swap=args.augmentation_random_swap,
          augmentation_random_crop_type=args.augmentation_random_crop_type,
          augmentation_random_crop_to_shape=args.augmentation_random_crop_to_shape,
          augmentation_random_remove_points=args.augmentation_random_remove_points,
          augmentation_random_brightness=args.augmentation_random_brightness,
          augmentation_random_contrast=args.augmentation_random_contrast,
          augmentation_random_saturation=args.augmentation_random_saturation,
          # Loss function settings
          w_stereo=args.w_stereo,
          w_monocular=args.w_monocular,
          w_color=args.w_color,
          w_structure=args.w_structure,
          w_sparse_depth=args.w_sparse_depth,
          w_ensemble_depth=args.w_ensemble_depth,
          w_ensemble_temperature=args.w_ensemble_temperature,
          w_sparse_select_ensemble=args.w_sparse_select_ensemble,
          sparse_select_dilate_kernel_size=args.sparse_select_dilate_kernel_size,
          w_smoothness=args.w_smoothness,
          w_weight_decay_depth=args.w_weight_decay_depth,
          w_weight_decay_pose=args.w_weight_decay_pose,
          loss_func_ensemble_depth=args.loss_func_ensemble_depth,
          epoch_pose_for_ensemble=args.epoch_pose_for_ensemble,
          ensemble_method=args.ensemble_method,
          # Evaluation settings
          min_evaluate_depth=args.min_evaluate_depth,
          max_evaluate_depth=args.max_evaluate_depth,
          # Checkpoint settings
          checkpoint_path=args.checkpoint_path,
          n_checkpoint=args.n_checkpoint,
          n_summary=args.n_summary,
          n_summary_display=args.n_summary_display,
          validation_start_step=args.validation_start_step,
          depth_model_restore_path=args.depth_model_restore_path,
          pose_model_restore_path=args.pose_model_restore_path,
          # Hardware settings
          device=args.device,
          n_thread=args.n_thread)
