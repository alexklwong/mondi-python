#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_external_model.py \
--image_path testing/void/void_test_image_1500.txt \
--sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--intrinsics_path testing/void/void_test_intrinsics_1500.txt \
--ground_truth_path testing/void/void_test_ground_truth_1500.txt \
--model_name msg_chn \
--restore_path external_models/msg_chn/void/msg_chn-void1500.pth \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--output_path external_models/msg_chn/void1500 \
--device gpu
