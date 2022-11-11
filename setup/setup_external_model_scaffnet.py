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

# ScaffNet pretrained models
SCAFFNET_PRETRAINED_MODELS_DIRPATH = \
    os.path.join(PRETRAINED_MODELS_DIRPATH, 'scaffnet')

# ScaffNet pretrained SceneNet model
SCAFFNET_PRETRAINED_SCENENET_MODEL_DIRPATH = \
    os.path.join(SCAFFNET_PRETRAINED_MODELS_DIRPATH, 'scenenet')

SCAFFNET_PRETRAINED_SCENENET_MODEL_URL = \
    GOOGLE_DRIVE_BASE_URL.format('1oMlxelSYQwYHtHKSN4iR0e0PFcdKAmTx')

SCAFFNET_PRETRAINED_SCENENET_MODEL_FILENAME = 'scaffnet-scenenet.pth'
SCAFFNET_PRETRAINED_SCENENET_MODEL_FILEPATH = \
    os.path.join(SCAFFNET_PRETRAINED_SCENENET_MODEL_DIRPATH, SCAFFNET_PRETRAINED_SCENENET_MODEL_FILENAME)

# FusionNet pretrained VOID model
FUSIONNET_PRETRAINED_VOID_MODEL_DIRPATH = \
    os.path.join(SCAFFNET_PRETRAINED_MODELS_DIRPATH, 'void')

FUSIONNET_PRETRAINED_VOID_MODEL_URL = \
    GOOGLE_DRIVE_BASE_URL.format('1QPPqxaek3TGhTc0E3V3UarwSAXvjFIVN')

FUSIONNET_PRETRAINED_VOID_MODEL_FILENAME = 'fusionnet_standalone-void1500.pth'
FUSIONNET_PRETRAINED_VOID_MODEL_FILEPATH = \
    os.path.join(FUSIONNET_PRETRAINED_VOID_MODEL_DIRPATH, FUSIONNET_PRETRAINED_VOID_MODEL_FILENAME)

# FusionNet pretrained NYUv2 model
FUSIONNET_PRETRAINED_NYU_V2_MODEL_DIRPATH = \
    os.path.join(SCAFFNET_PRETRAINED_MODELS_DIRPATH, 'nyu_v2')

FUSIONNET_PRETRAINED_NYU_V2_MODEL_URL = \
    GOOGLE_DRIVE_BASE_URL.format('1PshP7IBckeyds_sqvEn-XrNYv1gKcP4L')

FUSIONNET_PRETRAINED_NYU_V2_MODEL_FILENAME = 'fusionnet_standalone-nyu_v2.pth'
FUSIONNET_PRETRAINED_NYU_V2_MODEL_FILEPATH = \
    os.path.join(FUSIONNET_PRETRAINED_NYU_V2_MODEL_DIRPATH, FUSIONNET_PRETRAINED_NYU_V2_MODEL_FILENAME)


def setup_scaffnet_model():

    # Download pretrained model
    dirpaths = [
        SCAFFNET_PRETRAINED_MODELS_DIRPATH,
        SCAFFNET_PRETRAINED_SCENENET_MODEL_DIRPATH,
        FUSIONNET_PRETRAINED_VOID_MODEL_DIRPATH,
        FUSIONNET_PRETRAINED_NYU_V2_MODEL_DIRPATH
    ]

    for dirpath in dirpaths:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    if not os.path.exists(SCAFFNET_PRETRAINED_SCENENET_MODEL_FILEPATH):
        print('Downloading {} to {}'.format(
            SCAFFNET_PRETRAINED_SCENENET_MODEL_FILENAME, SCAFFNET_PRETRAINED_SCENENET_MODEL_FILEPATH))

        gdown.download(SCAFFNET_PRETRAINED_SCENENET_MODEL_URL, SCAFFNET_PRETRAINED_SCENENET_MODEL_FILEPATH, quiet=False)
    else:
        print('Found {} at {}'.format(
            SCAFFNET_PRETRAINED_SCENENET_MODEL_FILENAME, SCAFFNET_PRETRAINED_SCENENET_MODEL_FILEPATH))

    if not os.path.exists(FUSIONNET_PRETRAINED_VOID_MODEL_FILEPATH):
        print('Downloading {} to {}'.format(
            FUSIONNET_PRETRAINED_VOID_MODEL_FILENAME, FUSIONNET_PRETRAINED_VOID_MODEL_FILEPATH))

        gdown.download(FUSIONNET_PRETRAINED_VOID_MODEL_URL, FUSIONNET_PRETRAINED_VOID_MODEL_FILEPATH, quiet=False)
    else:
        print('Found {} at {}'.format(
            FUSIONNET_PRETRAINED_VOID_MODEL_FILENAME, FUSIONNET_PRETRAINED_VOID_MODEL_FILEPATH))

    if not os.path.exists(FUSIONNET_PRETRAINED_NYU_V2_MODEL_FILEPATH):
        print('Downloading {} to {}'.format(
            FUSIONNET_PRETRAINED_NYU_V2_MODEL_FILENAME, FUSIONNET_PRETRAINED_NYU_V2_MODEL_FILEPATH))

        gdown.download(FUSIONNET_PRETRAINED_NYU_V2_MODEL_URL, FUSIONNET_PRETRAINED_NYU_V2_MODEL_FILEPATH, quiet=False)
    else:
        print('Found {} at {}'.format(
            FUSIONNET_PRETRAINED_NYU_V2_MODEL_FILENAME, FUSIONNET_PRETRAINED_NYU_V2_MODEL_FILEPATH))


if __name__ == "__main__":

    setup_scaffnet_model()
