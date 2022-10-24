#!bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_fusionnet.py \
--image_path testing/void/void_test_image_1500.txt \
--dense_depth_path testing/void/void_test_predict_depth_1500.txt \
--sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--ground_truth_path testing/void/void_test_ground_truth_1500.txt \
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
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--checkpoint_path \
trained_fusionnet/void1500/vgg08_co020_st080_sz100_sm100_tp010_lr0-5e5_5-1e4_10-5e5_15/evaluation_results/void1500 \
--restore_path \
trained_fusionnet/void1500/vgg08_co020_st080_sz100_sm100_tp010_lr0-5e5_5-1e4_10-5e5_15/depth_model-600.pth \
--save_outputs \
--keep_input_filenames \
--device gpu
