#!bin/bash

mkdir -p data

wget http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar -P data
wget http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_depth.tar -P data

mkdir data/vkitti_2.0.3_rgb
tar -xvf data/vkitti_2.0.3_rgb.tar -C data/vkitti_2.0.3_rgb

mkdir data/vkitti_2.0.3_depth
tar -xvf data/vkitti_2.0.3_depth.tar -C data/vkitti_2.0.3_depth