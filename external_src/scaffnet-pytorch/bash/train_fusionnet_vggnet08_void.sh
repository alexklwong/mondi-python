#!bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_fusionnet.py \
--train_images_path training/void/void_train_image_1500.txt \
--train_dense_depth_path training/void/void_train_predict_depth_1500.txt \
--train_sparse_depth_path training/void/void_train_sparse_depth_1500.txt \
--train_intrinsics_path training/void/void_train_intrinsics_1500.txt \
--val_image_path testing/void/void_test_image_1500.txt \
--val_dense_depth_path testing/void/void_test_predict_depth_1500.txt \
--val_sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--val_ground_truth_path testing/void/void_test_ground_truth_1500.txt \
--n_batch 8 \
--n_height 480 \
--n_width 640 \
--normalized_image_range 0 1 \
--outlier_removal_kernel_size 7 \
--outlier_removal_threshold 1.5 \
--encoder_type vggnet08 \
--n_filters_encoder_image 48 96 192 384 384 \
--n_filters_encoder_depth 16 32 64 128 128 \
--decoder_type multi-scale \
--n_filters_decoder 256 128 128 64 32 \
--scale_match_method local_scale \
--scale_match_kernel_size 5 \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--min_multiplier_depth 0.25 \
--max_multiplier_depth 4.00 \
--min_residual_depth -1000.0 \
--max_residual_depth 1000.0 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--learning_rates 5e-5 1e-4 5e-5 \
--learning_schedule 5 10 15 \
--augmentation_random_crop_type none \
--w_color 0.20 \
--w_structure 0.80 \
--w_sparse_depth 1.00 \
--w_smoothness 1.00 \
--w_prior_depth 0.10 \
--threshold_prior_depth 0.30 \
--w_weight_decay_depth 0.00 \
--w_weight_decay_pose 0.00 \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--n_summary 1000 \
--n_summary_display 4 \
--n_checkpoint 1000 \
--checkpoint_path \
trained_fusionnet/void1500/vgg08_co020_st080_sz100_sm100_tp010_lr0-5e5_5-1e4_10-5e5_15 \
--device cuda \
--n_thread 8 \
