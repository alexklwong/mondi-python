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
sys.path.insert(0, os.path.join('external_src', 'kbnet'))
sys.path.insert(0, os.path.join('external_src', 'kbnet', 'src'))
from kbnet_model import KBNetModel as KBNet
from net_utils import OutlierRemoval


class KBNetModel(object):
    '''
    Class for interfacing with KBNet model

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
                 dataset_name='kitti',
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 device=torch.device('cuda')):

        # Instantiate depth completion model
        if dataset_name == 'kitti':
            min_pool_sizes_sparse_to_dense_pool = [5, 7, 9, 11, 13]
            max_pool_sizes_sparse_to_dense_pool = [15, 17]
        elif dataset_name == 'void' or dataset_name == 'nyu_v2':
            min_pool_sizes_sparse_to_dense_pool = [15, 17]
            max_pool_sizes_sparse_to_dense_pool = [23, 27, 29]
        else:
            raise ValueError('Unsupported dataset settings: {}'.format(dataset_name))

        self.model = KBNet(
            input_channels_image=3,
            input_channels_depth=2,
            min_pool_sizes_sparse_to_dense_pool=min_pool_sizes_sparse_to_dense_pool,
            max_pool_sizes_sparse_to_dense_pool=max_pool_sizes_sparse_to_dense_pool,
            n_convolution_sparse_to_dense_pool=3,
            n_filter_sparse_to_dense_pool=8,
            n_filters_encoder_image=[48, 96, 192, 384, 384],
            n_filters_encoder_depth=[16, 32, 64, 128, 128],
            resolutions_backprojection=[0, 1, 2, 3],
            n_filters_decoder=[256, 128, 128, 64, 12],
            deconv_type='up',
            weight_initializer='xavier_normal',
            activation_func='leaky_relu',
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth,
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

        image, \
            sparse_depth, \
            filtered_validity_map = self.transform_inputs(
                image=image,
                sparse_depth=sparse_depth)

        output_depth = self.model.forward(
            image=image,
            sparse_depth=sparse_depth,
            validity_map_depth=filtered_validity_map,
            intrinsics=intrinsics)

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

        # Filter validity map
        validity_map = torch.where(
            sparse_depth > 0,
            torch.ones_like(sparse_depth),
            sparse_depth)

        # Remove outlier points and update sparse depth and validity map
        filtered_sparse_depth, \
            filtered_validity_map = self.outlier_removal.remove_outliers(
                sparse_depth=sparse_depth,
                validity_map=validity_map)

        return image, sparse_depth, filtered_validity_map

    def compute_loss(self,
                     image0,
                     image1,
                     image2,
                     output_depth0,
                     filtered_sparse_depth0,
                     filtered_validity_map_depth0,
                     intrinsics,
                     pose01,
                     pose02,
                     w_color=0.15,
                     w_structure=0.95,
                     w_sparse_depth=0.60,
                     w_smoothness=0.04):
        '''
        Computes loss function
        l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm}

        Arg(s):
            image0 : torch.Tensor[float32]
                N x 3 x H x W image at time step t
            image1 : torch.Tensor[float32]
                N x 3 x H x W image at time step t-1
            image2 : torch.Tensor[float32]
                N x 3 x H x W image at time step t+1
            output_depth0 : torch.Tensor[float32]
                N x 1 x H x W output depth at time t
            filtered_sparse_depth0 : torch.Tensor[float32]
                N x 1 x H x W sparse depth at time t
            filtered_validity_map_depth0 : torch.Tensor[float32]
                N x 1 x H x W validity map of sparse depth at time t
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix
            pose01 : torch.Tensor[float32]
                N x 4 x 4 relative pose from image at time t to t-1
            pose02 : torch.Tensor[float32]
                N x 4 x 4 relative pose from image at time t to t+1
            w_color : float
                weight of color consistency term
            w_structure : float
                weight of structure consistency term (SSIM)
            w_sparse_depth : float
                weight of sparse depth consistency term
            w_smoothness : float
                weight of local smoothness term
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        loss, loss_info = self.model.compute_loss(
            image0=image0,
            image1=image1,
            image2=image2,
            output_depth0=output_depth0,
            sparse_depth0=filtered_sparse_depth0,
            validity_map_depth0=filtered_validity_map_depth0,
            intrinsics=intrinsics,
            pose01=pose01,
            pose02=pose02,
            w_color=w_color,
            w_structure=w_structure,
            w_sparse_depth=w_sparse_depth,
            w_smoothness=w_smoothness)

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return self.model.parameters()

    def train(self):
        '''
        Sets model to training mode
        '''

        self.model.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.model.eval()

    def to(self, device):
        '''
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''

        self.device = device
        self.model.to(device)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        # KBNet already calls data_parallel() in constructor
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

        _, optimizer = self.model.restore_model(restore_path, optimizer)

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

        self.model.save_model(checkpoint_path, step, optimizer)
