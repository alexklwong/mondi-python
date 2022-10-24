# Batch settings
N_BATCH                             = 8
N_HEIGHT                            = 196
N_WIDTH                             = 640
N_CHANNEL                           = 3

# ScaffNet encoder settings
ENCODER_TYPE_AVAILABLE_SCAFFNET     = ['vggnet08',
                                       'resnet18',
                                       'spatial_pyramid_pool',
                                       'outlier_removal']
ENCODER_TYPE_SCAFFNET               = ['vggnet08']
N_FILTERS_ENCODER_SCAFFNET          = [16, 32, 48, 64, 96]
SPATIAL_PYRAMID_POOL_SIZES          = [5, 7, 9, 11]
N_CONV_SPATIAL_PYRAMID_POOL         = 2
N_FILTER_SPATIAL_PYRAMID_POOL       = 8

# ScaffNet decoder settings
DECODER_TYPE_AVAILABLE_SCAFFNET     = ['multi-scale', 'uncertainty']
DECONV_TYPE_AVAILABLE_SCAFFNET      = ['transpose', 'up']
OUTPUT_FUNC_AVAILABLE_SCAFFNET      = ['linear',
                                       'sigmoid',
                                       'linear-upsample',
                                       'sigmoid-upsample']
DECODER_TYPE_SCAFFNET               = ['multi-scale']
N_FILTERS_DECODER_SCAFFNET          = [96, 96, 96, 64, 32]
DECONV_TYPE_SCAFFNET                = 'up'
OUTPUT_FUNC_SCAFFNET                = 'sigmoid'

# FusionNet encoder settings
ENCODER_TYPE_AVAILABLE_FUSIONNET    = ['vggnet08',
                                       'resnet18',
                                       'conv0']
INPUT_TYPE_AVAILABLE_FUSIONNET     = ['input_depth',
                                      'sparse_depth',
                                      'validity_map']
ENCODER_TYPE_FUSIONNET              = ['vggnet08']
INPUT_TYPE_FUSIONNET                = ['input_depth', 'sparse_depth']
N_FILTERS_ENCODER_IMAGE_FUSIONNET   = [48, 96, 192, 384, 384]
N_FILTERS_ENCODER_DEPTH_FUSIONNET   = [16, 32, 64, 128, 128]

# FusionNet decoder settings
DECODER_TYPE_AVAILABLE_FUSIONNET    = ['multi-scale']
OUTPUT_TYPE_AVAILABLE_FUSIONNET     = ['multiplier',
                                       'residual',
                                       'mapping',
                                       'clamp']
DECONV_TYPE_AVAILABLE_FUSIONNET     = ['transpose', 'up']
OUTPUT_FUNC_AVAILABLE_FUSIONNET     = ['linear', 'sigmoid']
DECODER_TYPE_FUSIONNET              = ['multi-scale']
OUTPUT_TYPE_FUSIONNET               = ['multiplier', 'residual']
N_FILTERS_DECODER_FUSIONNET         = [256, 128, 128, 64, 8]
DECONV_TYPE_FUSIONNET               = 'up'
OUTPUT_FUNC_MULTIPLIER_FUSIONNET    = 'linear'
OUTPUT_FUNC_RESIDUAL_FUSIONNET      = 'sigmoid'

# Weight settings
ACTIVATION_FUNC_AVAILABLE           = ['relu',
                                       'leaky_relu',
                                       'elu']
WEIGHT_INITIALIZER_AVAILABLE        = ['kaiming_normal',
                                       'kaiming_uniform',
                                       'xavier_normal',
                                       'xavier_uniform']
WEIGHT_INITIALIZER                  = 'kaiming_uniform'
ACTIVATION_FUNC                     = 'leaky_relu'
USE_BATCH_NORM                      = False

# General training settings
N_EPOCH                             = 60
LEARNING_RATES                      = [1.00e-4, 0.50e-4, 0.25e-4]
LEARNING_SCHEDULE                   = [20, 40]
USE_AUGMENT                         = False

# ScaffNet training settings
LOSS_FUNC_AVAILABLE_SCAFFNET        = ['supervised_l1',
                                       'supervised_l1_normalized',
                                       'weight_decay']
LOSS_FUNC_SCAFFNET                  = ['supervised_l1', 'weight_decay']
W_SUPERVISED                        = 1.00
W_WEIGHT_DECAY                      = 0.00
W_COLOR                             = 0.20
W_STRUCTURE                         = 0.80
W_SPARSE_DEPTH                      = 0.20
W_SMOOTHNESS                        = 0.01
W_PRIOR_DEPTH                       = 0.10
THRESHOLD_PRIOR_DEPTH               = 0.30
W_WEIGHT_DECAY_DEPTH                = 0.00
W_WEIGHT_DECAY_POSE                 = 1e-4

# Depth range settings
EPSILON                             = 1e-8
OUTLIER_REMOVAL_METHOD_AVAILABLE    = ['remove', 'set_to_min']
OUTLIER_REMOVAL_METHOD              = 'remove'
OUTLIER_REMOVAL_KERNEL_SIZE         = 7
OUTLIER_REMOVAL_THRESHOLD           = 1.5
SCALE_MATCH_METHOD_AVAILABLE        = ['replace',
                                       'local_scale',
                                       'local_scale_replace',
                                       'none']
SCALE_MATCH_METHOD                  = 'local_scale'
SCALE_MATCH_KERNEL_SIZE             = 3
CAP_DATASET_METHOD_AVAILABLE        = ['remove', 'set_to_max']
CAP_DATASET_METHOD                  = 'remove'
MIN_DATASET_DEPTH                   = 1e-3
MAX_DATASET_DEPTH                   = 655.0
MIN_PREDICT_DEPTH                   = 1e-3
MAX_PREDICT_DEPTH                   = 655.0
MIN_MULTIPLIER_DEPTH                = 0.25
MAX_MULTIPLIER_DEPTH                = 4.00
MIN_RESIDUAL_DEPTH                  = -80.0
MAX_RESIDUAL_DEPTH                  = 80.0
MIN_EVALUATE_DEPTH                  = 1e-3
MAX_EVALUATE_DEPTH                  = 655.0

# Rotation parameterization
POSE_TYPE_AVAILABLE                 = ['forward', 'backward']
ROTATION_PARAM_AVAILABLE            = ['euler', 'exponential', 'axis']
POSE_TYPE                           = ['forward', 'backward']
ROTATION_PARAM                      = 'axis'

# Checkpoint settings
N_DISPLAY                           = 4
N_SUMMARY                           = 1000
N_CHECKPOINT                        = 5000
CHECKPOINT_PATH                     = ''
RESTORE_PATH                        = ''

# Hardware settings
DEVICE                              = 'cuda'
CUDA                                = 'cuda'
CPU                                 = 'cpu'
GPU                                 = 'gpu'
N_THREAD                            = 8
