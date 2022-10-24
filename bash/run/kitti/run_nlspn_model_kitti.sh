#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_external_model.py \
--image_path validation/kitti/kitti_val_image.txt \
--sparse_depth_path validation/kitti/kitti_val_sparse_depth.txt \
--intrinsics_path validation/kitti/kitti_val_intrinsics.txt \
--ground_truth_path validation/kitti/kitti_val_ground_truth.txt \
--model_name nlspn \
--restore_path external_models/nlspn/kitti/NLSPN_KITTI_DC.pt \
--min_predict_depth 0.0 \
--max_predict_depth 100.0 \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 100.0 \
--output_path external_models/nlspn/kitti \
--device gpu
