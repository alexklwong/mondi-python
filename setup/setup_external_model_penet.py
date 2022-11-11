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

# PENET pretrained models
PENET_PRETRAINED_MODELS_DIRPATH = \
    os.path.join(PRETRAINED_MODELS_DIRPATH, 'penet')

# PENET pretrained KITTI model
PENET_PRETRAINED_KITTI_MODEL_DIRPATH = \
    os.path.join(PENET_PRETRAINED_MODELS_DIRPATH, 'kitti')

PENET_PRETRAINED_KITTI_MODEL_URL = \
    GOOGLE_DRIVE_BASE_URL.format('1RDdKlKJcas-G5OA49x8OoqcUDiYYZgeM')

PENET_PRETRAINED_KITTI_MODEL_FILENAME = 'pe.pth.tar'
PENET_PRETRAINED_KITTI_MODEL_FILEPATH = \
    os.path.join(PENET_PRETRAINED_KITTI_MODEL_DIRPATH, PENET_PRETRAINED_KITTI_MODEL_FILENAME)

# PENET pretrained VOID model
PENET_PRETRAINED_VOID_MODEL_DIRPATH = \
    os.path.join(PENET_PRETRAINED_MODELS_DIRPATH, 'void')

PENET_PRETRAINED_VOID_MODEL_URL = \
    GOOGLE_DRIVE_BASE_URL.format('11u_OffVLvH0J7tgk74OxVA15O4PbLct9')

PENET_PRETRAINED_VOID_MODEL_FILENAME = 'penet-void1500.pth'
PENET_PRETRAINED_VOID_MODEL_FILEPATH = \
    os.path.join(PENET_PRETRAINED_VOID_MODEL_DIRPATH, PENET_PRETRAINED_VOID_MODEL_FILENAME)


def setup_penet_model():

    # Download pretrained model
    dirpaths = [
        PENET_PRETRAINED_MODELS_DIRPATH,
        PENET_PRETRAINED_KITTI_MODEL_DIRPATH,
        PENET_PRETRAINED_VOID_MODEL_DIRPATH
    ]

    for dirpath in dirpaths:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    if not os.path.exists(PENET_PRETRAINED_KITTI_MODEL_FILEPATH):
        print('Downloading {} to {}'.format(
            PENET_PRETRAINED_KITTI_MODEL_FILENAME, PENET_PRETRAINED_KITTI_MODEL_FILEPATH))

        gdown.download(PENET_PRETRAINED_KITTI_MODEL_URL, PENET_PRETRAINED_KITTI_MODEL_FILEPATH, quiet=False)
    else:
        print('Found {} at {}'.format(
            PENET_PRETRAINED_KITTI_MODEL_FILENAME, PENET_PRETRAINED_KITTI_MODEL_FILEPATH))

    if not os.path.exists(PENET_PRETRAINED_VOID_MODEL_FILEPATH):
        print('Downloading {} to {}'.format(
            PENET_PRETRAINED_VOID_MODEL_FILENAME, PENET_PRETRAINED_VOID_MODEL_FILEPATH))

        gdown.download(PENET_PRETRAINED_VOID_MODEL_URL, PENET_PRETRAINED_VOID_MODEL_FILEPATH, quiet=False)
    else:
        print('Found {} at {}'.format(
            PENET_PRETRAINED_VOID_MODEL_FILENAME, PENET_PRETRAINED_VOID_MODEL_FILEPATH))


if __name__ == "__main__":

    setup_penet_model()
