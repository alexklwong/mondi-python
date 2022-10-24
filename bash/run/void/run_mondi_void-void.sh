#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
DENSITY=1500

python src/run_mondi.py \
--image_path testing/void/void_test_image_${DENSITY}.txt \
--sparse_depth_path testing/void/void_test_sparse_depth_${DENSITY}.txt \
--intrinsics_path testing/void/void_test_intrinsics_${DENSITY}.txt \
--ground_truth_path testing/void/void_test_ground_truth_${DENSITY}.txt \
--input_types image sparse_depth validity_map \
--input_channels_image 3 \
--input_channels_depth 2 \
--normalized_image_range 0 1 \
--outlier_removal_kernel_size 7 \
--outlier_removal_threshold 1.5 \
--min_pool_sizes_sparse_to_dense_pool 15 17 19 21 23 \
--max_pool_sizes_sparse_to_dense_pool 27 29 \
--n_convolution_sparse_to_dense_pool 3  \
--n_filter_sparse_to_dense_pool 8 \
--encoder_type kbnet \
--n_filters_encoder_image 48 96 192 384 384 \
--n_filters_encoder_depth 16 32 64 128 128 \
--n_convolutions_encoder 1 1 1 2 2 \
--resolutions_backprojection 0 1 2 3 \
--resolutions_depthwise_separable_encoder 4 5 \
--decoder_type kbnet \
--n_filters_decoder 256 128 128 64 12 \
--n_resolution_decoder 1 \
--resolutions_depthwise_separable_decoder -1 \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--depth_model_restore_path pretrained_models/void/mondi-void.pth \
--output_path model_output/void/evaluation_results/void_validation \
--device gpu
