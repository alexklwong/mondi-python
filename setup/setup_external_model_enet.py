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

GOOGLE_DRIVE_BASE_URL = 'https://drive.google.com/uc?id={}'

# ENET pretrained models
ENET_PRETRAINED_MODELS_DIRPATH = \
    os.path.join(PRETRAINED_MODELS_DIRPATH, 'enet')

# ENET pretrained KITTI model
ENET_PRETRAINED_KITTI_MODEL_DIRPATH = \
    os.path.join(ENET_PRETRAINED_MODELS_DIRPATH, 'kitti')

ENET_PRETRAINED_KITTI_MODEL_URL = \
    GOOGLE_DRIVE_BASE_URL.format('1TRVmduAnrqDagEGKqbpYcKCT307HVQp1')

ENET_PRETRAINED_KITTI_MODEL_FILENAME = 'e.pth.tar'
ENET_PRETRAINED_KITTI_MODEL_FILEPATH = \
    os.path.join(ENET_PRETRAINED_KITTI_MODEL_DIRPATH, ENET_PRETRAINED_KITTI_MODEL_FILENAME)

# ENET pretrained VOID model
ENET_PRETRAINED_VOID_MODEL_DIRPATH = \
    os.path.join(ENET_PRETRAINED_MODELS_DIRPATH, 'void')

ENET_PRETRAINED_VOID_MODEL_URL = \
    GOOGLE_DRIVE_BASE_URL.format('1lbdIwz4ya14NHTLjo9FPIswVi-KJr54J')

ENET_PRETRAINED_VOID_MODEL_FILENAME = 'enet-void1500.pth'
ENET_PRETRAINED_VOID_MODEL_FILEPATH = \
    os.path.join(ENET_PRETRAINED_VOID_MODEL_DIRPATH, ENET_PRETRAINED_VOID_MODEL_FILENAME)


def setup_enet_model():

    # Download pretrained model
    dirpaths = [
        ENET_PRETRAINED_MODELS_DIRPATH,
        ENET_PRETRAINED_KITTI_MODEL_DIRPATH,
        ENET_PRETRAINED_VOID_MODEL_DIRPATH
    ]

    for dirpath in dirpaths:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    if not os.path.exists(ENET_PRETRAINED_KITTI_MODEL_FILEPATH):
        print('Downloading {} to {}'.format(
            ENET_PRETRAINED_KITTI_MODEL_FILENAME, ENET_PRETRAINED_KITTI_MODEL_FILEPATH))

        gdown.download(ENET_PRETRAINED_KITTI_MODEL_URL, ENET_PRETRAINED_KITTI_MODEL_FILEPATH, quiet=False)
    else:
        print('Found {} at {}'.format(
            ENET_PRETRAINED_KITTI_MODEL_FILENAME, ENET_PRETRAINED_KITTI_MODEL_FILEPATH))

    if not os.path.exists(ENET_PRETRAINED_VOID_MODEL_FILEPATH):
        print('Downloading {} to {}'.format(
            ENET_PRETRAINED_VOID_MODEL_FILENAME, ENET_PRETRAINED_VOID_MODEL_FILEPATH))

        gdown.download(ENET_PRETRAINED_VOID_MODEL_URL, ENET_PRETRAINED_VOID_MODEL_FILEPATH, quiet=False)
    else:
        print('Found {} at {}'.format(
            ENET_PRETRAINED_VOID_MODEL_FILENAME, ENET_PRETRAINED_VOID_MODEL_FILEPATH))


if __name__ == "__main__":

    setup_enet_model()
