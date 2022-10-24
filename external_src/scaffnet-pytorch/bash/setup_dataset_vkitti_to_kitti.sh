#!bin/bash

export CUDA_VISIBLE_DEVICES=0

python setup/setup_dataset_vkitti_to_kitti.py \
--restore_path scaffnet_models/kitti/res18-xnorm-lrelu-bn-pool5-7-9-11_conv3-8-max_lr1e4-5e5-2e5_l1norm100_wd1e4_max100-120_16-32-48-64-96-gt/model-250000.pth \
--encoder_type resnet18 spatial_pyramid_pool outlier_removal \
--n_filters_encoder 16 32 48 64 96 \
--decoder_type multi-scale \
--n_filters_decoder 96 96 96 64 32 \
--deconv_type up \
--spatial_pyramid_pool_sizes 5 7 9 11 \
--n_conv_spatial_pyramid_pool 3 \
--n_filter_spatial_pyramid_pool 8 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--output_func sigmoid \
--use_batch_norm \
--min_predict_depth 1.5 \
--max_predict_depth 120.0 \
--device gpu
