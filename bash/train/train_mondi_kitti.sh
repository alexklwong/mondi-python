#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_mondi.py \
--train_image0_path \
    training/kitti/kitti_train_clean_image0.txt \
--train_image1_path \
    training/kitti/kitti_train_clean_image1.txt \
--train_sparse_depth0_path \
    training/kitti/kitti_train_clean_sparse_depth0.txt \
--train_sparse_depth1_path \
    training/kitti/kitti_train_clean_sparse_depth1.txt \
--train_ground_truth0_path \
    training/kitti/kitti_train_clean_ground_truth0.txt \
--train_ground_truth1_path \
    training/kitti/kitti_train_clean_ground_truth1.txt \
--train_teacher_output0_paths \
    training/kitti/kitti_train_clean_teacher_output0-nlspn.txt \
    training/kitti/kitti_train_clean_teacher_output0-penet.txt \
    training/kitti/kitti_train_clean_teacher_output0-enet.txt \
--train_teacher_output1_paths \
    training/kitti/kitti_train_clean_teacher_output1-nlspn.txt \
    training/kitti/kitti_train_clean_teacher_output1-penet.txt \
    training/kitti/kitti_train_clean_teacher_output1-enet.txt \
--train_intrinsics0_path \
    training/kitti/kitti_train_clean_instrinsics0.txt \
--train_intrinsics1_path \
    training/kitti/kitti_train_clean_instrinsics1.txt \
--train_focal_length_baseline0_path \
    training/kitti/kitti_train_clean_focal_length_baseline0.txt \
--train_focal_length_baseline1_path \
    training/kitti/kitti_train_clean_focal_length_baseline1.txt \
--val_image_path \
    validation/kitti/kitti_val_image.txt \
--val_sparse_depth_path \
    validation/kitti/kitti_val_sparse_depth.txt \
--val_intrinsics_path \
    validation/kitti/kitti_val_intrinsics.txt \
--val_ground_truth_path \
    validation/kitti/kitti_val_ground_truth.txt \
--n_batch 8 \
--n_height 320 \
--n_width 768 \
--input_types image sparse_depth validity_map \
--input_channels_image 3 \
--input_channels_depth 2 \
--normalized_image_range 0 1 \
--outlier_removal_kernel_size 7 \
--outlier_removal_threshold 1.5 \
--min_pool_sizes_sparse_to_dense_pool 5 7 9 11 13 \
--max_pool_sizes_sparse_to_dense_pool 15 17 \
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
--min_predict_depth 1.5 \
--max_predict_depth 100.0 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--supervision_types stereo monocular \
--learning_rates_depth 5e-4 2e-4 5e-5 2e-5 5e-5 2e-5 \
--learning_schedule_depth 30 50 90 100 120 200 \
--learning_rates_pose 5e-4 2e-4 5e-5 2e-5 5e-5 2e-5 \
--learning_schedule_pose 30 50 90 100 120 200 \
--augmentation_probabilities 1.00 0.5 \
--augmentation_schedule 100 200 \
--augmentation_random_crop_type horizontal vertical \
--augmentation_random_swap \
--augmentation_random_remove_points 0.20 0.70 \
--augmentation_random_brightness 0.80 1.20 \
--augmentation_random_contrast 0.80 1.20 \
--augmentation_random_saturation 0.80 1.20 \
--w_stereo 1.00 \
--w_monocular 1.00 \
--w_color 0.15 \
--w_structure 0.85 \
--w_sparse_depth 0.00 \
--w_ensemble_depth 1.00 \
--w_ensemble_temperature 0.10 \
--w_smoothness 0.00 \
--w_weight_decay_depth 0.00 \
--w_weight_decay_pose 0.00 \
--loss_func_ensemble_depth l1 \
--epoch_pose_for_ensemble 10 \
--ensemble_method mondi \
--w_sparse_select_ensemble 0.001 \
--sparse_select_dilate_kernel_size 3 \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 100.0 \
--n_summary 10000 \
--n_checkpoint 1000 \
--validation_start 1000 \
--checkpoint_path \
    trained_mondi/kitti/mondi_model \
--device gpu \
--n_thread 6
