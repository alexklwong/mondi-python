'''
Authors:
Tian Yu Liu <tianyu@cs.ucla.edu>
Parth Agrawal <parthagrawal24@ucla.edu>
Allison Chen <allisonchen2@ucla.edu>
Alex Wong <alex.wong@yale.edu>

If you use this code, please cite the following paper:
T.Y. Liu, P. Agrawal, A. Chen, B.W. Hong, and A. Wong. Monitored Distillation for Positive Congruent Depth Completion.
https://arxiv.org/abs/2203.16034

@inproceedings{liu2022monitored,
  title={Monitored distillation for positive congruent depth completion},
  author={Liu, Tian Yu and Agrawal, Parth and Chen, Allison and Hong, Byung-Woo and Wong, Alex},
  booktitle={European Conference on Computer Vision},
  year={2022},
  organization={Springer}
}
'''

import os, sys
import torch
sys.path.insert(0, os.path.join('external_src'))
sys.path.insert(0, os.path.join('external_src', 'scaffnet-pytorch'))
sys.path.insert(0, os.path.join('external_src', 'scaffnet-pytorch', 'src'))
from scaffnet_model import ScaffNetModel as ScaffNet
from fusionnet_model import FusionNetModel as FusionNet
from net_utils import OutlierRemoval


class FusionNetModel(object):
    '''
    Class for interfacing with FusionNet model

    Arg(s):
        dataset_name : str
            model for a given dataset
        min_predict_depth : float
            minimum value of predicted depth
        max_predict_depth : flaot
            maximum value of predicted depth
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 dataset_name='void',
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 device=torch.device('cuda')):

        # Instantiate depth completion model
        if dataset_name == 'kitti':
            pass
        elif dataset_name == 'void' or dataset_name == 'nyu_v2':
            max_pool_sizes_spatial_pyramid_pool = [13, 17, 19, 21, 25]
            n_convolution_spatial_pyramid_pool = 3
            n_filter_spatial_pyramid_pool = 8
            encoder_type_scaffnet = ['vggnet08', 'spatial_pyramid_pool', 'batch_norm']
            n_filters_encoder_scaffnet = [16, 32, 64, 128, 256]
            decoder_type_scaffnet = ['multi-scale', 'batch_norm']
            n_filters_decoder_scaffnet = [256, 128, 128, 64, 32]
            min_predict_depth_scaffnet = 0.1
            max_predict_depth_scaffnet = 10.0

            encoder_type_fusionnet = ['vggnet08']
            n_filters_encoder_image_fusionnet = [48, 96, 192, 384, 384]
            n_filters_encoder_depth_fusionnet = [16, 32, 64, 128, 128]
            decoder_type_fusionnet = ['multi-scale']
            n_filters_decoder_fusionnet = [256, 128, 128, 64, 32]
            scale_match_method_fusionnet = 'local_scale'
            scale_match_kernel_size_fusionnet = 5
            min_predict_depth_fusionnet = min_predict_depth
            max_predict_depth_fusionnet = max_predict_depth
            min_multiplier_depth_fusionnet = 0.25
            max_multiplier_depth_fusionnet = 4.00
            min_residual_depth_fusionnet = -1000.0
            max_residual_depth_fusionnet = 1000.0
        else:
            raise ValueError('Unsupported dataset settings: {}'.format(dataset_name))

        # Build ScaffNet
        self.scaffnet_model = ScaffNet(
            max_pool_sizes_spatial_pyramid_pool=max_pool_sizes_spatial_pyramid_pool,
            n_convolution_spatial_pyramid_pool=n_convolution_spatial_pyramid_pool,
            n_filter_spatial_pyramid_pool=n_filter_spatial_pyramid_pool,
            encoder_type=encoder_type_scaffnet,
            n_filters_encoder=n_filters_encoder_scaffnet,
            decoder_type=decoder_type_scaffnet,
            n_filters_decoder=n_filters_decoder_scaffnet,
            weight_initializer='xavier_normal',
            activation_func='leaky_relu',
            min_predict_depth=min_predict_depth_scaffnet,
            max_predict_depth=max_predict_depth_scaffnet,
            device=device)

        # Build FusionNet
        self.fusionnet_model = FusionNet(
            encoder_type=encoder_type_fusionnet,
            n_filters_encoder_image=n_filters_encoder_image_fusionnet,
            n_filters_encoder_depth=n_filters_encoder_depth_fusionnet,
            decoder_type=decoder_type_fusionnet,
            n_filters_decoder=n_filters_decoder_fusionnet,
            scale_match_method=scale_match_method_fusionnet,
            scale_match_kernel_size=scale_match_kernel_size_fusionnet,
            min_predict_depth=min_predict_depth_fusionnet,
            max_predict_depth=max_predict_depth_fusionnet,
            min_multiplier_depth=min_multiplier_depth_fusionnet,
            max_multiplier_depth=max_multiplier_depth_fusionnet,
            min_residual_depth=min_residual_depth_fusionnet,
            max_residual_depth=max_residual_depth_fusionnet,
            weight_initializer='xavier_normal',
            activation_func='leaky_relu',
            device=device)

        self.outlier_removal = OutlierRemoval(
            kernel_size=7,
            threshold=1.5)

        # Move to device
        self.device = device
        self.to(self.device)
        self.eval()

    def forward(self, image, sparse_depth, intrinsics):
        '''
        Forwards inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 intrinsic camera calibration matrix
        Returns:
            torch.Tensor[float32] : N x 1 x H x W dense depth map
        '''

        # Forward through ScaffNet
        input_depth = self.scaffnet_model.forward(sparse_depth)

        if 'uncertainty' in self.scaffnet_model.decoder_type:
            input_depth = input_depth[:, 0:1, :, :]

        image, \
            filtered_sparse_depth, \
            filtered_validity_map = self.transform_inputs(
                image=image,
                sparse_depth=sparse_depth)

        # Forward through FusionNet
        output_depth = self.fusionnet_model.forward(
            image=image,
            input_depth=input_depth,
            sparse_depth=filtered_sparse_depth)

        return output_depth

    def transform_inputs(self, image, sparse_depth):
        '''
        Transforms the input based on any required preprocessing step

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
        Returns:
            torch.Tensor[float32] : N x 3 x H x W image
            torch.Tensor[float32] : N x 1 x H x W sparse depth map
            torch.Tensor[float32] : N x 1 x H x W validity map
        '''

        image = image / 255.0

        # Validity map is where sparse depth is available
        validity_map = torch.where(
            sparse_depth > 0,
            torch.ones_like(sparse_depth),
            sparse_depth)

        # Remove outlier points and update sparse depth and validity map
        filtered_sparse_depth, \
            filtered_validity_map = self.outlier_removal.remove_outliers(
                sparse_depth=sparse_depth,
                validity_map=validity_map)

        return image, filtered_sparse_depth, filtered_validity_map

    def compute_loss(self,
                     output_depth0,
                     sparse_depth0,
                     validity_map0,
                     input_depth0,
                     image0,
                     image1,
                     image2,
                     pose0to1,
                     pose0to2,
                     intrinsics,
                     w_color,
                     w_structure,
                     w_sparse_depth,
                     w_smoothness,
                     w_prior_depth,
                     threshold_prior_depth):
        '''
        Computes loss function
        l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm} + w_{tp}l_{tp}

        Arg(s):
            output_depth0 : torch.Tensor[float32]
                N x 1 x H x W output depth for time t
            sparse_depth0 : torch.Tensor[float32]
                N x 1 x H x W sparse depth for time t
            validity_map0 : torch.Tensor[float32]
                N x 1 x H x W validity map of sparse depth for time t
            input_depth0 : torch.Tensor[float32]
                N x 1 x H x W input depth for time t
            image0 : torch.Tensor[float32]
                N x 3 x H x W time t image
            image1 : torch.Tensor[float32]
                N x 3 x H x W time t-1 image
            image2 : torch.Tensor[float32]
                N x 3 x H x W time t+1 image
            pose0to1 : torch.Tensor[float32]
                N x 4 x 4 transformation matrix from time t to t-1
            pose0to2 : torch.Tensor[float32]
                N x 4 x 4 transformation matrix from time t to t+1
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix
            w_color : float
                weight of color consistency term
            w_structure : float
                weight of structure consistency term (SSIM)
            w_sparse_depth : float
                weight of sparse depth consistency term
            w_smoothness : float
                weight of local smoothness term
            w_prior_depth : float
                weight of prior depth consistency term
            threshold_prior_depth : float
                threshold to start using prior depth term
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        # Compute loss function
        loss, loss_info = self.fusionnet_model.compute_loss(
            output_depth0=output_depth0,
            sparse_depth0=sparse_depth0,
            validity_map0=validity_map0,
            input_depth0=input_depth0,
            image0=image0,
            image1=image1,
            image2=image2,
            pose0to1=pose0to1,
            pose0to2=pose0to2,
            intrinsics=intrinsics,
            w_color=w_color,
            w_structure=w_structure,
            w_sparse_depth=w_sparse_depth,
            w_smoothness=w_smoothness,
            w_prior_depth=w_prior_depth,
            threshold_prior_depth=threshold_prior_depth)

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return self.fusionnet_model.parameters()

    def train(self):
        '''
        Sets model to training mode
        '''

        self.scaffnet_model.train()
        self.fusionnet_model.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.scaffnet_model.eval()
        self.fusionnet_model.eval()

    def to(self, device):
        '''
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''

        self.device = device
        self.scaffnet_model.to(device)
        self.fusionnet_model.to(device)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        pass

    def restore_model(self, restore_path, optimizer=None):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_path : str
                path to model weights
            optimizer : torch.optim
                optimizer
        '''

        _, optimizer = self.fusionnet_model.restore_model(
            checkpoint_path=restore_path,
            optimizer=optimizer,
            scaffnet_model=self.scaffnet_model)

        return optimizer

    def save_model(self, checkpoint_path, step, optimizer):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''

        self.fusionnet_model.save_model(
            checkpoint_path=checkpoint_path,
            step=step,
            optimizer=optimizer,
            scaffnet_model=self.fusionnet_model)
