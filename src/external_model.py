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


class ExternalModel(object):
    '''
    Wrapper class for all external depth completion models

    Arg(s):
        model_name : str
            depth completion model to use
        min_predict_depth : float
            minimum depth to predict
        max_predict_depth : float
            maximum depth to predict
        use_pretrained : bool
            if set, then load pretrained for NLSPN
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 model_name,
                 min_predict_depth,
                 max_predict_depth,
                 use_pretrained=False,
                 device=torch.device('cuda')):

        self.model_name = model_name
        self.device = device

        if model_name == 'nlspn':
            from nlspn_model import NLSPNModel
            self.model = NLSPNModel(
                device=device,
                max_depth=max_predict_depth,
                use_pretrained=use_pretrained)
        elif model_name == 'enet':
            from enet_model import ENetModel
            self.model = ENetModel(
                device=device,
                max_depth=max_predict_depth)
        elif model_name == 'penet':
            from penet_model import PENetModel
            self.model = PENetModel(
                device=device,
                max_depth=max_predict_depth)
        elif model_name == 'msg_chn':
            from msg_chn_model import MsgChnModel
            self.model = MsgChnModel(
                device=device,
                max_depth=max_predict_depth)
        elif 'scaffnet' in model_name:
            from scaffnet_models import ScaffNetModel

            if 'vkitti' in model_name:
                dataset_name = 'vkitti'
            elif 'scenenet' in model_name:
                dataset_name = 'scenenet'
            else:
                dataset_name = 'vkitti'

            self.model = ScaffNetModel(
                dataset_name=dataset_name,
                min_predict_depth=min_predict_depth,
                max_predict_depth=max_predict_depth,
                device=device)
        elif 'fusionnet' in model_name:
            from fusionnet_models import FusionNetModel

            if 'kitti' in model_name:
                dataset_name = 'kitti'
            elif 'void' in model_name:
                dataset_name = 'void'
            elif 'nyu_v2' in model_name:
                dataset_name = 'nyu_v2'
            else:
                dataset_name = 'kitti'

            self.model = FusionNetModel(
                dataset_name=dataset_name,
                min_predict_depth=min_predict_depth,
                max_predict_depth=max_predict_depth,
                device=device)
        elif 'kbnet' in model_name:
            from kbnet_models import KBNetModel

            if 'kitti' in model_name:
                dataset_name = 'kitti'
            elif 'void' in model_name:
                dataset_name = 'void'
            elif 'nyu_v2' in model_name:
                dataset_name = 'nyu_v2'
            else:
                dataset_name = 'kitti'

            self.model = KBNetModel(
                dataset_name=dataset_name,
                min_predict_depth=min_predict_depth,
                max_predict_depth=max_predict_depth,
                device=device)
        else:
            raise ValueError('Unsupported depth completion model: {}'.format(model_name))

    def forward(self, image, sparse_depth, intrinsics=None):
        '''
        Forwards inputs through network

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

        return self.model.forward(image, sparse_depth, intrinsics)

    def compute_loss(self, output_depth, ground_truth_depth):
        '''
        Call the model's compute loss function

        Currently only supports supervised methods (ENet, PENet, MSGCHN, NLSPN)
        Unsupervised methods have various more complex losses that is best trained through
        their repository

        Arg(s):
            output_depth : torch.Tensor[float32]
                N x 1 x H x W dense output depth already masked with validity map
            ground_truth_depth : torch.Tensor[float32]
                N x 1 x H x W ground_truth depth with only valid values
        Returns:
            float : loss averaged over the batch
        '''

        return self.model.compute_loss(
            output_depth=output_depth,
            ground_truth_depth=ground_truth_depth)

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

        self.model.data_parallel()

    def restore_model(self, restore_path, optimizer=None):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_path : str
                path to model weights
            optimizer : torch.optimizer or None
                current optimizer
        Returns:
            torch.optimizer or None if no optimizer is passed in
        '''

        return self.model.restore_model(restore_path, optimizer)

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
