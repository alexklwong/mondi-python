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

# MSG_CHN pretrained models
MSG_CHN_PRETRAINED_MODELS_DIRPATH = \
    os.path.join(PRETRAINED_MODELS_DIRPATH, 'msg_chn')

# MSG_CHN pretrained KITTI models
MSG_CHN_PRETRAINED_KITTI_MODEL_DIRPATH = \
    os.path.join(MSG_CHN_PRETRAINED_MODELS_DIRPATH, 'kitti')

MSG_CHN_PRETRAINED_KITTI_MODEL_URL = \
    GOOGLE_DRIVE_BASE_URL.format('15u4MP3y4MtTk2ile-bwX8Ff2qdufCgxf')

MSG_CHN_PRETRAINED_KITTI_MODEL_FILENAME = 'final.pth.tar'
MSG_CHN_PRETRAINED_KITTI_MODEL_FILEPATH = \
    os.path.join(MSG_CHN_PRETRAINED_KITTI_MODEL_DIRPATH, MSG_CHN_PRETRAINED_KITTI_MODEL_FILENAME)

# MSG_CHN pretrained VOID models
MSG_CHN_PRETRAINED_VOID_MODEL_DIRPATH = \
    os.path.join(MSG_CHN_PRETRAINED_MODELS_DIRPATH, 'void')

MSG_CHN_PRETRAINED_VOID_MODEL_URL = \
    GOOGLE_DRIVE_BASE_URL.format('1QS2IpDX58EJyFm93KJWo-6SuNfMOrfHh')

MSG_CHN_PRETRAINED_VOID_MODEL_FILENAME = 'msg_chn-void1500.pth'
MSG_CHN_PRETRAINED_VOID_MODEL_FILEPATH = \
    os.path.join(MSG_CHN_PRETRAINED_VOID_MODEL_DIRPATH, MSG_CHN_PRETRAINED_VOID_MODEL_FILENAME)


def setup_msg_chn_model():

    # Download pretrained model
    dirpaths = [
        MSG_CHN_PRETRAINED_MODELS_DIRPATH,
        MSG_CHN_PRETRAINED_KITTI_MODEL_DIRPATH,
        MSG_CHN_PRETRAINED_VOID_MODEL_DIRPATH
    ]

    for dirpath in dirpaths:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    if not os.path.exists(MSG_CHN_PRETRAINED_KITTI_MODEL_FILEPATH):
        print('Downloading {} to {}'.format(
            MSG_CHN_PRETRAINED_KITTI_MODEL_FILENAME, MSG_CHN_PRETRAINED_KITTI_MODEL_FILEPATH))

        gdown.download(MSG_CHN_PRETRAINED_KITTI_MODEL_URL, MSG_CHN_PRETRAINED_KITTI_MODEL_FILEPATH, quiet=False)
    else:
        print('Found {} at {}'.format(
            MSG_CHN_PRETRAINED_KITTI_MODEL_FILENAME, MSG_CHN_PRETRAINED_KITTI_MODEL_FILEPATH))

    if not os.path.exists(MSG_CHN_PRETRAINED_VOID_MODEL_FILEPATH):
        print('Downloading {} to {}'.format(
            MSG_CHN_PRETRAINED_VOID_MODEL_FILENAME, MSG_CHN_PRETRAINED_VOID_MODEL_FILEPATH))

        gdown.download(MSG_CHN_PRETRAINED_VOID_MODEL_URL, MSG_CHN_PRETRAINED_VOID_MODEL_FILEPATH, quiet=False)
    else:
        print('Found {} at {}'.format(
            MSG_CHN_PRETRAINED_VOID_MODEL_FILENAME, MSG_CHN_PRETRAINED_VOID_MODEL_FILEPATH))


if __name__ == "__main__":

    setup_msg_chn_model()
