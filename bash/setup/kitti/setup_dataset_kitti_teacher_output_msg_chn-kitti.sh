#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python setup/setup_dataset_kitti_teacher_output.py \
--external_models \
    msg_chn \
--external_models_restore_paths \
    external_models/msg_chn/kitti/final.pth.tar \
--min_predict_depth 0.0 \
--max_predict_depth 100.0
