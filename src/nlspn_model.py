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

import os, sys, argparse
import torchvision
import loss_utils
from data_utils import inpainting
sys.path.insert(0, os.path.join('external_src', 'NLSPN'))
sys.path.insert(0, os.path.join('external_src', 'NLSPN', 'src'))
sys.path.insert(0, os.path.join('external_src', 'NLSPN', 'src', 'model'))
from nlspnmodel import NLSPNModel as NLSPNBaseModel
import torch


class NLSPNModel(object):
    '''
    Class for interfacing with NLSPN model

    Arg(s):
        device : torch.device
            device to run model on
        max_depth : float
            value to clamp ground truths to in computing loss
        use_pretrained : bool
            if set, then configure using legacy settings
    '''

    def __init__(self, device=torch.device('cuda'), max_depth=100.0, use_pretrained=False):

        # Settings to reproduce NLSPN numbers on KITTI
        args = argparse.Namespace(
            affinity='TGASS',
            affinity_gamma=0.5,
            conf_prop=True,
            from_scratch=True,
            legacy=use_pretrained,
            lr=0.001,
            max_depth=max_depth,
            network='resnet34',
            preserve_input=True,
            prop_kernel=3,
            prop_time=18,
            test_only=True)
        # args = argparse.Namespace(
        #     affinity='TGASS',
        #     affinity_gamma=0.5,
        #     conf_prop=True,
        #     from_scratch=True,
        #     legacy=use_pretrained,
        #     lr=0.001,
        #     max_depth=max_depth,
        #     network='resnet34',
        #     preserve_input=True,
        #     prop_kernel=3,
        #     prop_time=256,
        #     test_only=True)

        # Instantiate depth completion model
        self.model = NLSPNBaseModel(args)
        self.use_pretrained = use_pretrained
        self.max_depth = max_depth

        # Move to device
        self.device = device
        self.to(self.device)

    def forward(self, image, sparse_depth, intrinsics=None):
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

        # Transform inputs
        image, sparse_depth, = self.transform_inputs(image, sparse_depth)
        # Forward through the model
        sample = {
            'rgb': image,
            'dep': sparse_depth
        }

        output = self.model.forward(sample)

        output_depth = output['pred']

        # Fill in any holes with inpainting
        if self.use_pretrained and not self.model.training:
            output_depth = output_depth.detach().cpu().numpy()
            output_depth = inpainting(output_depth)
            output_depth = torch.from_numpy(output_depth).to(self.device)

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

        image = image / 256.0

        for batch in range(image.shape[0]):

            image[batch, ...] = torchvision.transforms.functional.normalize(
                image[batch, ...],
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))

        return image, sparse_depth

    def compute_loss(self,
                     output_depth,
                     ground_truth_depth,
                     l1_weight=1.0,
                     l2_weight=1.0):
        '''
        Compute loss as NLSPN does: 1.0 * L1 + 1.0 * L2

        Arg(s):
            output_depth : torch.Tensor[float32]
                N x 1 x H x W dense output depth already masked with validity map
            ground_truth_depth : torch.Tensor[float32]
                N x 2 x H x W ground_truth depth and ground truth validity map
            l1_weight : float
                weight of l1 loss
            l2_weight : float
                weight of l2 loss
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''
        # They clamp their predictions
        output_depth = torch.clamp(output_depth, min=0, max=self.max_depth)
        ground_truth_depth = torch.clamp(ground_truth_depth, min=0, max=self.max_depth)

        # Obtain valid values
        validity_map = torch.unsqueeze(ground_truth_depth[:, 1, :, :], dim=1)
        ground_truth = torch.unsqueeze(ground_truth_depth[:, 0, :, :], dim=1)

        # Compute individual losses
        l1_loss = loss_utils.l1_loss(ground_truth, output_depth, validity_map)
        l2_loss = loss_utils.l2_loss(ground_truth, output_depth, validity_map)
        loss = l1_weight * l1_loss + l2_weight * l2_loss

        # Store loss info
        loss_info = {
            'l1_loss': l1_loss,
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

        parameters = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                parameters.append(param)

        parameters = torch.nn.ParameterList(parameters)

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
        self.model.load_state_dict(checkpoint['net'], strict=True)

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
                'net': self.model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_step': step
            }
        else:
            checkpoint = {
                'net': self.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_step': step
            }
        torch.save(checkpoint, checkpoint_path)
