#!bin/bash

export CUDA_VISIBLE_DEVICES=0

python setup/setup_dataset_scenenet_to_void.py \
--restore_path \
trained_scaffnet/scenenet/vgg08unc_8x240x320_pmin01max100_dmin00max100_lr0-1e4_4_5e5_10/model-230000.pth \
--max_pool_sizes_spatial_pyramid_pool 13 17 19 21 25 \
--n_convolution_spatial_pyramid_pool 3 \
--n_filter_spatial_pyramid_pool 8 \
--encoder_type vggnet08 spatial_pyramid_pool batch_norm \
--n_filters_encoder 16 32 64 128 256 \
--decoder_type multi-scale uncertainty batch_norm \
--n_filters_decoder 256 128 128 64 32 \
--min_predict_depth 0.10 \
--max_predict_depth 10.0 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--paths_only \
--device gpu

