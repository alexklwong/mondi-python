#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python setup/setup_dataset_void_teacher_output.py \
--external_models \
    fusionnet_nyu_v2 \
--external_models_restore_paths \
    external_models/scaffnet/nyu_v2/fusionnet_standalone-nyu_v2.pth \
--min_predict_depth 0.1 \
--max_predict_depth 8.0
