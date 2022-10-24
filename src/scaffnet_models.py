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


class ScaffNetModel(object):
    '''
    Class for interfacing with ScaffNet model

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
                 dataset_name='scenenet',
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 device=torch.device('cuda')):

        # Instantiate depth completion model
        if dataset_name == 'vkitti':
            pass
        elif dataset_name == 'scenenet':
            max_pool_sizes_spatial_pyramid_pool = [13, 17, 19, 21, 25]
            n_convolution_spatial_pyramid_pool = 3
            n_filter_spatial_pyramid_pool = 8
            encoder_type_scaffnet = ['vggnet08', 'spatial_pyramid_pool', 'batch_norm']
            n_filters_encoder_scaffnet = [16, 32, 64, 128, 256]
            decoder_type_scaffnet = ['multi-scale', 'uncertainty', 'batch_norm']
            n_filters_decoder_scaffnet = [256, 128, 128, 64, 32]
            min_predict_depth_scaffnet = min_predict_depth
            max_predict_depth_scaffnet = max_predict_depth
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
        output_depth = self.scaffnet_model.forward(sparse_depth)

        if 'uncertainty' in self.scaffnet_model.decoder_type:
            output_depth = output_depth[:, 0:1, :, :]

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
        '''

        return image, sparse_depth

    def compute_loss(self,
                     loss_func,
                     target_depth,
                     output_depth,
                     output_uncertainty=None,
                     w_supervised=1.00):
        '''
        Computes loss function

        Arg(s):
            loss_func : list[str]
                loss functions to minimize
            target_depth : torch.Tensor[float32]
                N x 1 x H x W groundtruth target depth
            output_depth : torch.Tensor[float32]
                N x 1 x H x W output depth
            output_uncertainty : torch.Tensor[float32]
                N x 1 x H x W uncertainty
            w_supervised : float
                weight of supervised loss
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : loss related infor
        '''

        # Compute loss function
        loss, loss_info = self.scaffnet_model.compute_loss(
            loss_func=loss_func,
            target_depth=target_depth,
            output_depth=output_depth,
            output_uncertainty=output_uncertainty,
            w_supervised=w_supervised)

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return self.scaffnet_model.parameters()

    def train(self):
        '''
        Sets model to training mode
        '''

        self.scaffnet_model.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.scaffnet_model.eval()

    def to(self, device):
        '''
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''

        self.device = device
        self.scaffnet_model.to(device)

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

        _, optimizer = self.scaffnet_model.restore_model(
            checkpoint_path=restore_path,
            optimizer=optimizer)

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

        self.scaffnet_model.save_model(
            checkpoint_path=checkpoint_path,
            step=step,
            optimizer=optimizer)
