#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_mondi.py \
--train_image0_path training/void/void_train_image_1500.txt \
--train_sparse_depth0_path training/void/void_train_sparse_depth_1500.txt \
--train_teacher_output0_paths \
    training/void/void_train_teacher_output_1500-kbnet_void.txt \
    training/void/void_train_teacher_output_1500-fusionnet_void.txt \
    training/void/void_train_teacher_output_1500-scaffnet_scenenet.txt \
    training/void/void_train_teacher_output_1500-nlspn.txt \
    training/void/void_train_teacher_output_1500-msg_chn.txt \
    training/void/void_train_teacher_output_1500-penet.txt \
    training/void/void_train_teacher_output_1500-enet.txt \
--train_intrinsics0_path training/void/void_train_intrinsics_1500.txt \
--train_ground_truth0_path \
    training/void/void_train_ground_truth_1500.txt \
--val_image_path testing/void/void_test_image_1500.txt \
--val_sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--val_intrinsics_path testing/void/void_test_intrinsics_1500.txt \
--val_ground_truth_path testing/void/void_test_ground_truth_1500.txt \
--n_batch 8 \
--n_height 448 \
--n_width 576 \
--supervision_types monocular \
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
--resolutions_backprojection 0 1 2 3 \
--resolutions_depthwise_separable_encoder 4 5 \
--decoder_type kbnet \
--n_resolution_decoder 1 \
--n_filters_decoder 256 128 128 64 12 \
--n_convolutions_encoder 1 1 1 2 2 \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--learning_rates_depth 5e-4 2e-4 5e-5 \
--learning_schedule_depth 15 20 50 \
--learning_rates_pose 5e-4 2e-4 5e-5 \
--learning_schedule_pose 15 20 50 \
--augmentation_probabilities 1.00 \
--augmentation_schedule -1 \
--augmentation_random_crop_type horizontal vertical \
--augmentation_random_crop_to_shape -1 -1 \
--augmentation_random_brightness 0.80 1.20 \
--augmentation_random_contrast 0.80 1.20 \
--augmentation_random_saturation 0.80 1.20 \
--augmentation_random_remove_points 0.60 0.95 \
--w_color 0.15 \
--w_structure 0.85 \
--w_sparse_depth 0.00 \
--w_ensemble_depth 1.00 \
--w_ensemble_temperature 0.1 \
--w_smoothness 0.00 \
--w_weight_decay_depth 0.00 \
--w_weight_decay_pose 0.00 \
--loss_func_ensemble_depth l1 \
--epoch_pose_for_ensemble 10 \
--ensemble_method mondi \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--n_summary 500000 \
--n_checkpoint 1000 \
--validation_start 20000 \
--checkpoint_path trained_mondi/void1500/mondi_model \
--device gpu \
--n_thread 8
