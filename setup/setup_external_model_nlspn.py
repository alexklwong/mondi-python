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

import os, gdown


# General global constants
PRETRAINED_MODELS_DIRPATH = 'external_models'
REPO_DIRPATH = os.getcwd()

GOOGLE_DRIVE_BASE_URL = 'https://drive.google.com/uc?id={}'

# NLSPN directory paths
NLSPN_DCN_DIRPATH = \
    os.path.join('external_src', 'NLSPN', 'src', 'model', 'deformconv')
NLSPN_DCN_BUILD_DIRPATH = \
    os.path.join(NLSPN_DCN_DIRPATH, 'build', 'lib.linux-x86_64-{}.{}')
NLSPN_DCN_PYFUNC_DIRPATH = \
    os.path.join(NLSPN_DCN_BUILD_DIRPATH, 'functions')
NLPSN_DCN_PYMODULE_DIRPATH = \
    os.path.join(NLSPN_DCN_BUILD_DIRPATH, 'modules')

# NLSPN build paths
NLSPN_DCN_BINARY_FILEPATH = \
    os.path.join(NLSPN_DCN_BUILD_DIRPATH, 'DCN.cpython-{}{}m-x86_64-linux-gnu.so')

NLSPN_DCN_DEFORM_CONV_PYFUNC_FILEPATH = \
    os.path.join(NLSPN_DCN_PYFUNC_DIRPATH, 'deform_conv_func.py')
NLSPN_DCN_DEFORM_POOL_PYFUNC_FILEPATH = \
    os.path.join(NLSPN_DCN_PYFUNC_DIRPATH, 'deform_psroi_pooling_func.py')
NLSPN_DCN_MODULATED_DEFORM_CONV_PYFUNC_FILEPATH = \
    os.path.join(NLSPN_DCN_PYFUNC_DIRPATH, 'modulated_deform_conv_func.py')

NLSPN_DCN_DEFORM_CONV_PYMODULE_FILEPATH = \
    os.path.join(NLPSN_DCN_PYMODULE_DIRPATH, 'deform_conv.py')
NLSPN_DCN_DEFORM_POOL_PYMODULE_FILEPATH = \
    os.path.join(NLPSN_DCN_PYMODULE_DIRPATH, 'deform_psroi_pooling.py')
NLSPN_DCN_MODULATED_DEFORM_CONV_PYMODULE_FILEPATH = \
    os.path.join(NLPSN_DCN_PYMODULE_DIRPATH, 'modulated_deform_conv.py')

NLSPN_DCN_BUILD_ITEMS = [
    NLSPN_DCN_BINARY_FILEPATH,
    NLSPN_DCN_DEFORM_CONV_PYFUNC_FILEPATH,
    NLSPN_DCN_DEFORM_POOL_PYFUNC_FILEPATH,
    NLSPN_DCN_MODULATED_DEFORM_CONV_PYFUNC_FILEPATH,
    NLSPN_DCN_DEFORM_CONV_PYMODULE_FILEPATH,
    NLSPN_DCN_DEFORM_POOL_PYMODULE_FILEPATH,
    NLSPN_DCN_MODULATED_DEFORM_CONV_PYMODULE_FILEPATH
]

# NLSPN pretrained model
NLSPN_PRETRAINED_MODELS_DIRPATH = \
    os.path.join(PRETRAINED_MODELS_DIRPATH, 'nlspn')

# NLSPN pretrained KITTI model
NLSPN_PRETRAINED_KITTI_MODEL_DIRPATH = \
    os.path.join(NLSPN_PRETRAINED_MODELS_DIRPATH, 'kitti')

NLSPN_PRETRAINED_KITTI_MODEL_URL = GOOGLE_DRIVE_BASE_URL.format('11by_1oglcncSHFeF3S2ldcysQYjyDcQh')

NLSPN_PRETRAINED_KITTI_MODEL_FILENAME = 'NLSPN_KITTI_DC.pt'

NLSPN_PRETRAINED_KITTI_MODEL_FILEPATH = \
    os.path.join(NLSPN_PRETRAINED_KITTI_MODEL_DIRPATH, NLSPN_PRETRAINED_KITTI_MODEL_FILENAME)

# NLSPN pretrained NYUv2 model
NLSPN_PRETRAINED_NYUV2_MODEL_DIRPATH = \
    os.path.join(NLSPN_PRETRAINED_MODELS_DIRPATH, 'nyu_v2')

NLSPN_PRETRAINED_NYUV2_MODEL_URL = GOOGLE_DRIVE_BASE_URL.format('1LTdZI36zdOeVmKRbG3orzvwXctg-OjGC')

NLSPN_PRETRAINED_NYUV2_MODEL_FILENAME = 'NLSPN_NYU.pt'

NLSPN_PRETRAINED_NYUV2_MODEL_FILEPATH = \
    os.path.join(NLSPN_PRETRAINED_NYUV2_MODEL_DIRPATH, NLSPN_PRETRAINED_NYUV2_MODEL_FILENAME)

# NLSPN pretrained VOID model
NLSPN_PRETRAINED_VOID_MODEL_DIRPATH = \
    os.path.join(NLSPN_PRETRAINED_MODELS_DIRPATH, 'void')

NLSPN_PRETRAINED_VOID_MODEL_URL = GOOGLE_DRIVE_BASE_URL.format('1LTdZI36zdOeVmKRbG3orzvwXctg-OjGC')

NLSPN_PRETRAINED_VOID_MODEL_FILENAME = 'nlspn-void1500.pth'

NLSPN_PRETRAINED_VOID_MODEL_FILEPATH = \
    os.path.join(NLSPN_PRETRAINED_VOID_MODEL_DIRPATH, NLSPN_PRETRAINED_VOID_MODEL_FILENAME)


def setup_nlspn_model():

    # Setup Deformable Convolution V2 module
    os.chdir(NLSPN_DCN_DIRPATH)
    os.system('bash make.sh')

    os.chdir(REPO_DIRPATH)

    # Download pretrained model
    dirpaths = [
        NLSPN_PRETRAINED_MODELS_DIRPATH,
        NLSPN_PRETRAINED_KITTI_MODEL_DIRPATH,
        NLSPN_PRETRAINED_NYUV2_MODEL_DIRPATH,
        NLSPN_PRETRAINED_VOID_MODEL_DIRPATH
    ]

    for dirpath in dirpaths:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    if not os.path.exists(NLSPN_PRETRAINED_KITTI_MODEL_FILEPATH):
        print('Downloading {} to {}'.format(
            NLSPN_PRETRAINED_KITTI_MODEL_FILENAME, NLSPN_PRETRAINED_KITTI_MODEL_FILEPATH))

        gdown.download(NLSPN_PRETRAINED_KITTI_MODEL_URL, NLSPN_PRETRAINED_KITTI_MODEL_FILEPATH, quiet=False)
    else:
        print('Found {} at {}'.format(
            NLSPN_PRETRAINED_KITTI_MODEL_FILENAME, NLSPN_PRETRAINED_KITTI_MODEL_FILEPATH))

    if not os.path.exists(NLSPN_PRETRAINED_NYUV2_MODEL_FILEPATH):
        print('Downloading {} to {}'.format(
            NLSPN_PRETRAINED_NYUV2_MODEL_FILENAME, NLSPN_PRETRAINED_NYUV2_MODEL_FILEPATH))

        gdown.download(NLSPN_PRETRAINED_NYUV2_MODEL_URL, NLSPN_PRETRAINED_NYUV2_MODEL_FILEPATH, quiet=False)
    else:
        print('Found {} at {}'.format(
            NLSPN_PRETRAINED_NYUV2_MODEL_FILENAME, NLSPN_PRETRAINED_NYUV2_MODEL_FILEPATH))

    if not os.path.exists(NLSPN_PRETRAINED_VOID_MODEL_FILEPATH):
        print('Downloading {} to {}'.format(
            NLSPN_PRETRAINED_VOID_MODEL_FILENAME, NLSPN_PRETRAINED_VOID_MODEL_FILEPATH))

        gdown.download(NLSPN_PRETRAINED_VOID_MODEL_URL, NLSPN_PRETRAINED_VOID_MODEL_FILEPATH, quiet=False)
    else:
        print('Found {} at {}'.format(
            NLSPN_PRETRAINED_VOID_MODEL_FILENAME, NLSPN_PRETRAINED_VOID_MODEL_FILEPATH))


if __name__ == "__main__":

    setup_nlspn_model()
