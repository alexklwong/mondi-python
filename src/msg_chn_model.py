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

import torch
import os, sys
import loss_utils
from net_utils import load_state_dict
sys.path.insert(0, os.path.join('external_src', 'MSG_CHN'))
sys.path.insert(0, os.path.join('external_src', 'MSG_CHN', 'workspace', 'exp_msg_chn'))
from network_exp_msg_chn import network


class MsgChnModel(object):
    '''
    Class for interfacing with MSGCHN model

    Arg(s):
        device : torch.device
            device to run model on
        max_depth : float
            value to clamp ground truths to in computing loss
    '''

    def __init__(self, device=torch.device('cuda'), max_depth=5.0):
        # Initialize model
        self.model = network()

        self.max_depth = max_depth

        # Move to device
        self.device = device
        self.to(self.device)

    def forward(self, image, sparse_depth, intrinsics):
        '''
        Forwards inputs through the network

        Arg(s):
            image : tensor[float32]
                N x 3 x H x W image
            sparse_depth : tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 intrinsic camera calibration matrix
        Returns:
            tensor[float32] : N x 1 x H x W input sparse depth map
        '''
        image, sparse_depth, intrinsics = self.transform_inputs(
            image,
            sparse_depth,
            intrinsics)

        n_height, n_width = image.shape[-2:]

        do_padding = False

        # Pad to width and height such that it is divisible by 16
        if n_height % 16 != 0:
            times = n_height // 16
            padding_top = (times + 1) * 16 - n_height
            do_padding = True
        else:
            padding_top = 0

        if n_width % 16 != 0:
            times = n_width // 16
            padding_right = (times + 1) * 16 - n_width
            do_padding = True
        else:
            padding_right = 0

        if do_padding:
            # Pad the images and expand at 0-th dimension to get batch
            image0 = torch.nn.functional.pad(
                image,
                (0, padding_right, padding_top, 0, 0, 0),
                mode='constant',
                value=0)

            sparse_depth0 = torch.nn.functional.pad(
                sparse_depth,
                (0, padding_right, padding_top, 0, 0, 0),
                mode='constant',
                value=0)

            image1 = torch.nn.functional.pad(
                image,
                (padding_right, 0, 0, padding_top, 0, 0),
                mode='constant',
                value=0)

            sparse_depth1 = torch.nn.functional.pad(
                sparse_depth,
                (padding_right, 0, 0, padding_top, 0, 0),
                mode='constant',
                value=0)

            image = torch.cat([image0, image1], dim=0)
            sparse_depth = torch.cat([sparse_depth0, sparse_depth1], dim=0)

        output, _, _ = self.model.forward(sparse_depth, image)

        if do_padding:
            output0, output1 = torch.chunk(output, chunks=2, dim=0)
            output0 = output0[:, :, padding_top:, :-padding_right]
            output1 = output1[:, :, :-padding_top, padding_right:]

            output = torch.cat([
                torch.unsqueeze(output0, dim=1),
                torch.unsqueeze(output1, dim=1)],
                dim=1)

            output = torch.mean(output, dim=1, keepdim=False)

        return output

    def transform_inputs(self, image, sparse_depth, intrinsics):
        '''
        Transforms the input based on any required preprocessing step
        Arg(s):
            image : tensor[float32]
                N x 3 x H x W image
            sparse_depth : tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 intrinsic camera calibration matrix
        Returns:
            tensor[float32] : N x 3 x H x W image
            tensor[float32] : N x 1 x H x W input sparse depth map
        '''

        # Normalization
        image = image / 255.0

        return image, sparse_depth, intrinsics

    def compute_loss(self,
                     output_depth,
                     ground_truth_depth):
        '''
        Compute L2 (MSE) Loss

        Arg(s):
            output_depth : torch.Tensor[float32]
                N x 1 x H x W dense output depth already masked with validity map
            ground_truth_depth : torch.Tensor[float32]
                N x 2 x H x W ground_truth depth and ground truth validity map
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        # Clamp ground truth values
        ground_truth_depth = torch.clamp(ground_truth_depth, min=0, max=self.max_depth)

        # Obtain valid values
        validity_map = torch.unsqueeze(ground_truth_depth[:, 1, :, :], dim=1)
        ground_truth = torch.unsqueeze(ground_truth_depth[:, 0, :, :], dim=1)

        # Compute loss
        loss = loss_utils.l2_loss(ground_truth, output_depth, validity_map)
        loss_info = { 'loss': loss }

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model
        Returns:
            list[torch.tensor[float32]] : list of parameters
        '''

        parameters = list(self.model.parameters())
        return parameters

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
        pass

    def save_model(self, checkpoint_path, step, optimizer):
        '''
        Saves weights to checkpoint
        Arg(s):
            checkpoint_path : str
                path to save checkpoint to
            step : int
                step number
            optimizer : torch.optimizer
                optimizer
        '''

        if isinstance(self.model, torch.nn.DataParallel):
            state_dict = {
                'net': self.model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }
        else:
            state_dict = {
                'net': self.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }

        torch.save(state_dict, checkpoint_path)

    def restore_model(self, restore_path, optimizer=None):
        '''
        Loads weights from checkpoint and loads and returns optimizer

        Arg(s):
            restore_path : str
                path to model weights
            optimizer : torch.optimizer or None
                current optimizer
        Returns:
            torch.optimizer if optimizer is passed in
        '''

        checkpoint_dict = torch.load(restore_path, map_location=self.device)
        module_state_dict = checkpoint_dict['net']

        self.model = load_state_dict(self.model, module_state_dict)

        if optimizer is not None and 'optimizer' in checkpoint_dict.keys():
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
            return optimizer
