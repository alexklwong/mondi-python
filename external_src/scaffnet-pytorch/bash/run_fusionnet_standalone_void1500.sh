#!bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_fusionnet_standalone.py \
--image_path testing/void/void_test_image_1500.txt \
--sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--ground_truth_path testing/void/void_test_ground_truth_1500.txt \
--normalized_image_range 0 1 \
--outlier_removal_kernel_size 7 \
--outlier_removal_threshold 1.5 \
--max_pool_sizes_spatial_pyramid_pool 13 17 19 21 25 \
--n_convolution_spatial_pyramid_pool 3 \
--n_filter_spatial_pyramid_pool 8 \
--encoder_type_scaffnet vggnet08 spatial_pyramid_pool batch_norm \
--n_filters_encoder_scaffnet 16 32 64 128 256 \
--decoder_type_scaffnet multi-scale uncertainty batch_norm \
--n_filters_decoder_scaffnet 256 128 128 64 32 \
--min_predict_depth_scaffnet 0.1 \
--max_predict_depth_scaffnet 10.0 \
--encoder_type_fusionnet vggnet08 \
--n_filters_encoder_image_fusionnet 48 96 192 384 384 \
--n_filters_encoder_depth_fusionnet 16 32 64 128 128 \
--decoder_type_fusionnet multi-scale \
--n_filters_decoder_fusionnet 256 128 128 64 32 \
--scale_match_method_fusionnet local_scale \
--scale_match_kernel_size_fusionnet 5 \
--min_predict_depth_fusionnet 0.1 \
--max_predict_depth_fusionnet 8.0 \
--min_multiplier_depth_fusionnet 0.25 \
--max_multiplier_depth_fusionnet 4.00 \
--min_residual_depth_fusionnet -1000.0 \
--max_residual_depth_fusionnet 1000.0 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--checkpoint_path \
    pretrained_models/void/evaluation_results/void1500/fusionnet-standalone \
--restore_path \
    pretrained_models/void/fusionnet_standalone-void1500.pth \
--keep_input_filenames \
--device gpu
