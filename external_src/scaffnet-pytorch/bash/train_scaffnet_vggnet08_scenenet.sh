#!bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_scaffnet.py \
--train_sparse_depth_path training/scenenet/scenenet_train_sparse_depth_corner.txt \
--train_ground_truth_path training/scenenet/scenenet_train_ground_truth_corner.txt \
--val_sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--val_ground_truth_path testing/void/void_test_ground_truth_1500.txt \
--n_batch 8 \
--n_height 240 \
--n_width 320 \
--cap_dataset_depth_method set_to_max \
--min_dataset_depth 0.0 \
--max_dataset_depth 10.0 \
--max_pool_sizes_spatial_pyramid_pool 13 17 19 21 25 \
--n_convolution_spatial_pyramid_pool 3 \
--n_filter_spatial_pyramid_pool 8 \
--encoder_type vggnet08 spatial_pyramid_pool batch_norm \
--n_filters_encoder 16 32 64 128 256 \
--decoder_type multi-scale uncertainty batch_norm \
--n_filters_decoder 256 128 128 64 32 \
--min_predict_depth 0.1 \
--max_predict_depth 10.0 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--learning_rates 1e-4 5e-5 \
--learning_schedule 4 10 \
--augmentation_random_crop_type none \
--loss_func supervised_l1_normalized \
--w_supervised 1.00 \
--w_weight_decay 0.00 \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--n_summary 5000 \
--n_checkpoint 5000 \
--checkpoint_path trained_scaffnet/scenenet/vgg08uncdepth_8x240x320_pmin01max100_dmin00max100_lr0-1e4_4_5e5_10 \
--device gpu \
--n_thread 8
