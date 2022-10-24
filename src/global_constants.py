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

EPSILON                                     = 1e-8

# Batch settings
N_BATCH                                     = 8
N_HEIGHT                                    = 320
N_WIDTH                                     = 768

# Input settings
INPUT_TYPES                                 = ['image', 'sparse_depth', 'filtered_validity_map']
INPUT_CHANNELS_IMAGE                        = 3
INPUT_CHANNELS_DEPTH                        = 2
NORMALIZED_IMAGE_RANGE                      = [0, 1]
OUTLIER_REMOVAL_KERNEL_SIZE                 = 7
OUTLIER_REMOVAL_THRESHOLD                   = 1.5

# Sparse to dense pool settings
MIN_POOL_SIZES_SPARSE_TO_DENSE_POOL         = [5, 7, 9, 11, 13]
MAX_POOL_SIZES_SPARSE_TO_DENSE_POOL         = [15, 17]
N_CONVOLUTION_SPARSE_TO_DENSE_POOL          = 3
N_FILTER_SPARSE_TO_DENSE_POOL               = 8

# Depth network settings
ENCODER_TYPE                                = ['resnet18']
N_FILTERS_ENCODER_IMAGE                     = [48, 96, 192, 384, 384]
N_FILTERS_ENCODER_DEPTH                     = [16, 32, 64, 128, 128]
N_CONVOLUTIONS_ENCODER                      = [1, 1, 1, 1, 1]
RESOLUTIONS_BACKPROJECTION                  = [0, 1, 2, 3]
RESOLUTIONS_DEPTHWISE_SEPARABLE_ENCODER     = [-1]
DECODER_TYPE                                = ['multiscale']
RESOLUTIONS_DEPTHWISE_SEPARABLE_DECODER     = [-1]
N_RESOLUTION_DECODER                        = 1
N_FILTERS_DECODER                           = [256, 128, 128, 64, 12]
DECONV_TYPE                                 = 'up'
MIN_PREDICT_DEPTH                           = 1.5
MAX_PREDICT_DEPTH                           = 100.0

# Weight settings
WEIGHT_INITIALIZER                          = 'xavier_normal'
ACTIVATION_FUNC                             = 'leaky_relu'
OUTPUT_FUNC                                 = 'linear'

# Training settings
SUPERVISION_TYPES                           = ['stereo']
LEARNING_RATES                              = [5e-5, 1e-4, 15e-5, 1e-4, 5e-5, 2e-5]
LEARNING_SCHEDULE                           = [2, 8, 20, 30, 45, 60]
AUGMENTATION_PROBABILITIES                  = [1.00, 0.50, 0.25]
AUGMENTATION_SCHEDULE                       = [50, 55, 60]
AUGMENTATION_RANDOM_CROP_TYPE               = ['horizontal', 'vertical', 'anchored', 'bottom']
AUGMENTATION_RANDOM_CROP_TO_SHAPE           = [-1, -1, -1, -1]
AUGMENTATION_RANDOM_FLIP_TYPE               = ['none']
AUGMENTATION_RANDOM_REMOVE_POINTS           = [0.60, 0.70]
AUGMENTATION_RANDOM_BRIGHTNESS              = [-1]
AUGMENTATION_RANDOM_CONTRAST                = [-1]
AUGMENTATION_RANDOM_SATURATION              = [-1]

# Loss function settings
W_STEREO                                    = 1.00
W_MONOCULAR                                 = 1.00
W_COLOR                                     = 0.15
W_STRUCTURE                                 = 0.95
W_SPARSE_DEPTH                              = 0.00
W_ENSEMBLE_DEPTH                            = 0.10
W_ENSEMBLE_TEMPERATURE                      = 0.00
W_SPARSE_SELECT_ENSEMBLE                    = 1.00
SPARSE_SELECT_DILATE_KERNEL_SIZE            = 3
W_SMOOTHNESS                                = 0.00
W_WEIGHT_DECAY_DEPTH                        = 0.00
W_WEIGHT_DECAY_POSE                         = 0.00
LOSS_FUNC_ENSEMBLE_DEPTH                    = 'l1'
EPOCH_POSE_FOR_ENSEMBLE                     = 10
ENSEMBLE_METHOD                             = 'mondi'

# Evaluation settings
MIN_EVALUATE_DEPTH                          = 0.00
MAX_EVALUATE_DEPTH                          = 100.0

# Checkpoint settings
CHECKPOINT_PATH                             = 'trained_kbnet'
N_CHECKPOINT                                = 5000
N_SUMMARY                                   = 5000
N_SUMMARY_DISPLAY                           = 4
VALIDATION_START_STEP                       = 200000
RESTORE_PATH                                = None

# Hardware settings
CUDA                                        = 'cuda'
CPU                                         = 'cpu'
GPU                                         = 'gpu'
DEVICE                                      = 'cuda'
DEVICE_AVAILABLE                            = [CPU, CUDA, GPU]
N_THREAD                                    = 8
