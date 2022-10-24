#!bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/run_scaffnet.py \
--sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--ground_truth_path testing/void/void_test_ground_truth_1500.txt \
--max_pool_sizes_spatial_pyramid_pool 13 17 19 21 25 \
--n_convolution_spatial_pyramid_pool 3 \
--n_filter_spatial_pyramid_pool 8 \
--encoder_type vggnet08 spatial_pyramid_pool batch_norm \
--n_filters_encoder 16 32 64 128 256 \
--decoder_type multi-scale batch_norm \
--n_filters_decoder 256 128 128 64 32 \
--min_predict_depth 0.10 \
--max_predict_depth 10.0 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--checkpoint_path \
    pretrained_models/void/evaluation_results/void1500/scaffnet \
--restore_path \
    pretrained_models/void/scaffnet-scenenet-v0.pth \
--keep_input_filenames \
--device gpu

