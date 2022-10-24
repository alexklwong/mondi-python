#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python setup/setup_dataset_void_teacher_output.py \
--external_models \
    penet \
--external_models_restore_paths \
    external_models/penet/void/penet-void1500.pth \
--min_predict_depth 0.1 \
--max_predict_depth 8.0
