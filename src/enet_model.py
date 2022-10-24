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
import loss_utils
sys.path.insert(0, os.path.join('external_src', 'PENet_ICRA2021'))
from model_modified import ENet
import CoordConv


class ENetModel(object):
    '''
    Class for interfacing with ENet model

    Arg(s):
        device : torch.device
            device to run model on
        max_depth : float
            value to clamp ground truths to in computing loss
    '''

    def __init__(self, device=torch.device('cuda'), max_depth=5.0):

        # Instantiate depth completion model
        network_model = 'e'
        convolutional_layer_encoding = 'xyz'
        dilation_rate = 2

        self.model = ENet(
            convolutional_layer_encoding=convolutional_layer_encoding,
            network_model=network_model,
            dilation_rate=dilation_rate)

        self.max_depth = max_depth

        # Move to device
        self.device = device
        self.to(self.device)

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

        _, _, og_height, og_width = image.shape

        image, sparse_depth, intrinsics = self.transform_inputs(image, sparse_depth, intrinsics)

        _, _, height, width = image.shape

        position = CoordConv.AddCoordsNp(height, width)
        position = torch.from_numpy(position.call())
        position = torch.unsqueeze(position, dim=0)
        position = position.permute(0, 3, 1, 2)
        position = position.to(self.device)

        position = position.repeat((image.shape[0], 1, 1, 1))

        self.rgb_depth, self.d_depth, output = self.model(image, sparse_depth, position, intrinsics)

        # Adds a bias
        output = output + 0.1

        if not self.model.training:
            final_output = self.take_avg(output, og_height, og_width)
        else:
            final_output = output

        return final_output

    def transform_inputs(self, image, sparse_depth, intrinsics):
        '''
        Transforms the input based on any required preprocessing step

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 intrinsic camera calibration matrix
        Returns:
            torch.Tensor[float32] : N x 3 x H x W image
            torch.Tensor[float32] : N x 1 x H x W sparse depth map
        '''

        _, c, h, w = image.shape

        if h % 32 == 0 and w % 32 == 0:
            return image, sparse_depth, intrinsics

        if not self.model.training:
            image_list = []
            sparse_depth_list = []
            intrinsics_list = []

            # Create top, bottom, left, right adjusted crops
            for i in range(4):

                if i == 0:
                    x_start = 0
                    y_start = 0
                elif i == 1:
                    x_start = w - 1216
                    y_start = h - 352
                elif i == 2:
                    x_start = 0
                    y_start = h - 352
                else:
                    x_start = w - 1216
                    y_start = 0

                x_end = x_start + 1216
                y_end = y_start + 352

                # Adjust intrinsics based on crop position
                crop_intrinsics = torch.tensor([[0.0, 0.0, -x_start],
                                                [0.0, 0.0, -y_start],
                                                [0.0, 0.0, 0.0]],
                                                dtype=torch.float32,
                                                device=intrinsics.device)

                image_cropped = image[:, :, y_start:y_end, x_start:x_end]
                sparse_depth_cropped = sparse_depth[:, :, y_start:y_end, x_start:x_end]
                intrinsics_cropped = intrinsics + crop_intrinsics

                image_list.append(image_cropped)
                sparse_depth_list.append(sparse_depth_cropped)
                intrinsics_list.append(intrinsics_cropped)

            # Concatenate along batch for a single forward pass
            image = torch.cat(image_list, dim=0)
            sparse_depth = torch.cat(sparse_depth_list, dim=0)
            intrinsics = torch.cat(intrinsics_list, dim=0)

        return image, sparse_depth, intrinsics

    def compute_loss(self,
                     output_depth,
                     ground_truth_depth):
        '''
        Compute L2 loss for ENet with weighing the rgb_depth and d_depth

        Arg(s):
            output_depth : torch.Tensor[float32]
                N x 1 x H x W dense output depth already masked with validity map
            ground_truth_depth : torch.Tensor[float32]
                N x 2 x H x W ground_truth depth and ground truth validity map
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        w_rgb_depth, w_d_depth = 0.05, 0.05

        # Clamp ground truth values
        ground_truth_depth = torch.clamp(ground_truth_depth, min=0, max=self.max_depth)

        # Obtain valid values
        validity_map = torch.unsqueeze(ground_truth_depth[:, 1, :, :], dim=1)
        ground_truth = torch.unsqueeze(ground_truth_depth[:, 0, :, :], dim=1)

        rgb_loss = loss_utils.l2_loss(ground_truth, self.rgb_depth, validity_map)
        d_loss = loss_utils.l2_loss(ground_truth, self.d_depth, validity_map)
        l2_loss = loss_utils.l2_loss(ground_truth, output_depth, validity_map)

        loss = (1 - w_rgb_depth - w_d_depth) * l2_loss + \
            w_rgb_depth * rgb_loss + \
            w_d_depth * d_loss

        loss_info = {
            'rgb_loss': rgb_loss,
            'd_loss': d_loss,
            'l2_loss': l2_loss,
            'loss': loss
        }

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
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

        self.model = torch.nn.DataParallel(self.model)

    def restore_model(self, restore_path, optimizer=None):
        '''
        Loads weights from checkpoint and loads and returns optimizer

        Arg(s):
            restore_path : str
                path to model weights
            optimizer : torch.optimizer or None
                current optimizer
        Returns:
            torch.optimizer or None if no optimizer is passed in
        '''

        checkpoint = torch.load(restore_path)

        self.model.load_state_dict(checkpoint['model'], strict=True)

        if optimizer is not None and 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
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

        if isinstance(self.model, torch.nn.DataParallel):
            checkpoint = {
                'model': self.model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_step': step
            }
        else:
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_step': step
            }
        torch.save(checkpoint, checkpoint_path)

    def take_avg(self, output, height, width):
        '''
        If validating by anchoring in each corner, compute average and output final prediction

        Arg(s):
            output : torch.Tensor[float32]
                N x 1 x out_H x out_W tensor; output of model
                if N == 1 then do not average
            height : int
                original image height
            width : int
                original image width

        Returns:
            torch.Tensor[float32] : N x 1 x H x W final output (N should be 1 in validation)
        '''

        n_batch, _, out_height, out_width = output.shape

        if n_batch == 1 and out_height == height and out_width == width:
            return output

        output_list = []
        validity_map_list = []

        for i in range(4):

            output_tmp = torch.zeros((1, height, width))
            validity_map = torch.zeros((1, height, width))

            if i == 0:
                x_start = 0
                y_start = 0
            elif i == 1:
                x_start = width - 1216
                y_start = height - 352
            elif i == 2:
                x_start = 0
                y_start = height - 352
            else:
                x_start = width - 1216
                y_start = 0

            x_end = x_start + 1216
            y_end = y_start + 352

            # Collect positions that have valid predictions
            output_tmp[:, y_start:y_end, x_start:x_end] = output[i]
            validity_map = torch.where(
                output_tmp > 0,
                torch.ones_like(validity_map),
                torch.zeros_like(validity_map))

            output_list.append(output_tmp)
            validity_map_list.append(validity_map)

        # Average over the overlaps for 4 x h x w
        output_full_size = torch.cat(output_list, dim=0)
        validity_maps = torch.cat(validity_map_list, dim=0)

        final_output = torch.sum(output_full_size, dim=0, keepdims=True) / torch.sum(validity_maps, dim=0, keepdims=True)

        return final_output
