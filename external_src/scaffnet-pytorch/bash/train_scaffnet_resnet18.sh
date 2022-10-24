#!bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_scaffnet.py \
--train_sparse_depth_path training/vkitti_train_sparse_depth.txt \
--train_ground_truth_path training/vkitti_train_semi_dense_depth.txt \
--val_sparse_depth_path validation/kitti_val_sparse_depth.txt \
--val_ground_truth_path validation/kitti_val_semi_dense_depth.txt \
--n_batch 8 \
--n_height 320 \
--n_width 768 \
--encoder_type resnet18 spatial_pyramid_pool \
--n_filters_encoder 16 32 64 96 128 \
--decoder_type multi-scale \
--n_filters_decoder 128 128 128 64 32 \
--deconv_type up \
--spatial_pyramid_pool_sizes 3 5 7 9 \
--n_conv_spatial_pyramid_pool 3 \
--n_filter_spatial_pyramid_pool 8 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--output_func sigmoid \
--use_batch_norm \
--n_epoch 5 \
--learning_rates 1e-4 5e-5 2e-5 \
--learning_schedule 2 4 \
--use_data_augmentation \
--n_random_shift_position 3 \
--loss_func supervised_l1_normalized weight_decay \
--w_supervised 1.00 \
--w_weight_decay 1e-4 \
--cap_dataset_method remove \
--min_dataset_depth 1e-3 \
--max_dataset_depth 100.0 \
--min_predict_depth 1.5 \
--max_predict_depth 100.0 \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 100.0 \
--n_summary 1000 \
--n_checkpoint 5000 \
--checkpoint_path trained_scaffnet_models/model \
--device gpu \
--n_thread 8
