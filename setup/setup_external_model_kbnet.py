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

import os, gdown, zipfile, shutil


# General global constants
PRETRAINED_MODELS_DIRPATH = 'external_models'

GOOGLE_DRIVE_BASE_URL = 'https://drive.google.com/uc?id={}'

# KBNet pretrained KITTI and VOID models
KBNET_PRETRAINED_MODELS_DIRPATH = \
    os.path.join(PRETRAINED_MODELS_DIRPATH, 'kbnet')

KBNET_PRETRAINED_MODEL_URL = \
    GOOGLE_DRIVE_BASE_URL.format('1C2RHo6E_Q8TzXN_h-GjrojJk4FYzQfRT')

KBNET_PRETRAINED_MODEL_FILENAME = 'pretrained_models.zip'
KBNET_PRETRAINED_MODEL_FILEPATH = \
    os.path.join(KBNET_PRETRAINED_MODELS_DIRPATH, KBNET_PRETRAINED_MODEL_FILENAME)

# KBNet pretrained NYUv2 model
KBNET_PRETRAINED_MODELS_DIRPATH = \
    os.path.join(KBNET_PRETRAINED_MODELS_DIRPATH, 'nyu_v2')

KBNET_PRETRAINED_NYU_V2_MODEL_URL = \
    GOOGLE_DRIVE_BASE_URL.format('1fvYWlKa-m4P-VQfqlz5LpUEwYdK86Fl6')

KBNET_PRETRAINED_NYU_V2_MODEL_FILENAME = 'kbnet-nyu_v2.pth'
KBNET_PRETRAINED_NYU_V2_MODEL_FILEPATH = \
    os.path.join(KBNET_PRETRAINED_MODELS_DIRPATH, KBNET_PRETRAINED_NYU_V2_MODEL_FILENAME)


def setup_kbnet_model():

    # Download pretrained model
    if not os.path.exists(KBNET_PRETRAINED_MODELS_DIRPATH):
        os.makedirs(KBNET_PRETRAINED_MODELS_DIRPATH)

    if not os.path.exists(KBNET_PRETRAINED_MODEL_FILEPATH):
        print('Downloading {} to {}'.format(
            KBNET_PRETRAINED_MODEL_FILENAME, KBNET_PRETRAINED_MODEL_FILEPATH))

        gdown.download(KBNET_PRETRAINED_MODEL_URL, KBNET_PRETRAINED_MODEL_FILEPATH, quiet=False)
    else:
        print('Found {} at {}'.format(
            KBNET_PRETRAINED_MODEL_FILENAME, KBNET_PRETRAINED_MODEL_FILEPATH))

    with zipfile.ZipFile(KBNET_PRETRAINED_MODEL_FILEPATH, 'r') as z:
        z.extractall(KBNET_PRETRAINED_MODELS_DIRPATH)

        shutil.move(
            os.path.join(KBNET_PRETRAINED_MODELS_DIRPATH, 'pretrained_models', 'kitti'),
            KBNET_PRETRAINED_MODELS_DIRPATH)

        shutil.move(
            os.path.join(KBNET_PRETRAINED_MODELS_DIRPATH, 'pretrained_models', 'void'),
            KBNET_PRETRAINED_MODELS_DIRPATH)

        os.rmdir(os.path.join(KBNET_PRETRAINED_MODELS_DIRPATH, 'pretrained_models'))

    if not os.path.exists(KBNET_PRETRAINED_MODELS_DIRPATH):
        os.makedirs(KBNET_PRETRAINED_MODELS_DIRPATH)

    if not os.path.exists(KBNET_PRETRAINED_NYU_V2_MODEL_FILEPATH):
        print('Downloading {} to {}'.format(
            KBNET_PRETRAINED_NYU_V2_MODEL_FILENAME, KBNET_PRETRAINED_NYU_V2_MODEL_FILEPATH))

        gdown.download(KBNET_PRETRAINED_NYU_V2_MODEL_URL, KBNET_PRETRAINED_NYU_V2_MODEL_FILEPATH, quiet=False)
    else:
        print('Found {} at {}'.format(
            KBNET_PRETRAINED_NYU_V2_MODEL_FILENAME, KBNET_PRETRAINED_NYU_V2_MODEL_FILEPATH))


if __name__ == "__main__":

    setup_kbnet_model()
