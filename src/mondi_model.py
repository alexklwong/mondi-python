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

import torch, torchvision
import log_utils
import loss_utils
import networks
import random


class MonitoredDistillationModel(object):
    '''
    Monitored distillation student model class

    Arg(s):
        encoder_type : list[str]
            encoder type (ex: ['kbnet', 'batch_norm'])
        input_channels_image : int
            number of channels in the image
        input_channels_depth : int
            number of channels in depth map
        n_filters_encoder_image : list[int]
            list of filters for each layer in image encoder
        n_filters_encoder_depth : list[int]
            list of filters for each layer in depth encoder
        n_convolutions_encoder : list[int]
            encoder convolutions for kbnet
        resolutions_backprojection : list[int]
            resolutions to do backprojection
        resolutions_depthwise_separable_encoder : list[int]
            resolutions to use depthwise separable convolution
        decoder_type : str
            decoder type
        n_resolution_decoder : int
            minimum resolution of multiscale outputs is 1/(2^n_resolution_decoder)
        n_filters_decoder : list[int]
            list of filters for each layer in decoder
        resolutions_depthwise_separable_decoder : list[int]
            resolutions to use depthwise separable convolution
        activation_func : str
            activation function for network
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        min_predict_depth : float
            minimum predicted depth
        max_predict_depth : float
            maximum predicted depth
        device : torch.device
            device for running model
    '''

    def __init__(self,
                 min_pool_sizes_sparse_to_dense_pool,
                 max_pool_sizes_sparse_to_dense_pool,
                 n_convolution_sparse_to_dense_pool,
                 n_filter_sparse_to_dense_pool,
                 encoder_type,
                 input_channels_image,
                 input_channels_depth,
                 n_filters_encoder_image,
                 n_filters_encoder_depth,
                 n_convolutions_encoder,
                 resolutions_backprojection,
                 resolutions_depthwise_separable_encoder,
                 decoder_type,
                 n_filters_decoder,
                 n_resolution_decoder,
                 resolutions_depthwise_separable_decoder,
                 activation_func,
                 weight_initializer,
                 min_predict_depth,
                 max_predict_depth,
                 device=torch.device('cuda')):

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.device = device

        # Build sparse to dense pooling
        self.sparse_to_dense_pool = networks.SparseToDensePool(
            input_channels=input_channels_depth,
            min_pool_sizes=min_pool_sizes_sparse_to_dense_pool,
            max_pool_sizes=max_pool_sizes_sparse_to_dense_pool,
            n_convolution=n_convolution_sparse_to_dense_pool,
            n_filter=n_filter_sparse_to_dense_pool,
            weight_initializer=weight_initializer,
            activation_func=activation_func)
        input_channels_depth = n_filter_sparse_to_dense_pool

        # Calculate number of channels in encoder combined for skip connections
        n_filters_encoder = [
            i + z
            for i, z in zip(n_filters_encoder_image, n_filters_encoder_depth)
        ]

        # Build encoder
        if 'resnet18' in encoder_type:
            self.encoder = networks.ResNetBasedEncoder(
                n_layer=18,
                input_channels_image=input_channels_image,
                input_channels_depth=input_channels_depth,
                n_filters_image=n_filters_encoder_image,
                n_filters_depth=n_filters_encoder_depth,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm='batch_norm' in encoder_type)

        elif 'kbnet' in encoder_type:
            n_convolutions_encoder_image = n_convolutions_encoder
            n_convolutions_encoder_depth = n_convolutions_encoder
            n_convolutions_encoder_fused = n_convolutions_encoder
            n_filters_encoder_fused = n_filters_encoder_image.copy()

            self.encoder = networks.KBNetEncoder(
                input_channels_image=input_channels_image,
                input_channels_depth=input_channels_depth,
                n_filters_image=n_filters_encoder_image,
                n_filters_depth=n_filters_encoder_depth,
                n_filters_fused=n_filters_encoder_fused,
                n_convolutions_image=n_convolutions_encoder_image,
                n_convolutions_depth=n_convolutions_encoder_depth,
                n_convolutions_fused=n_convolutions_encoder_fused,
                resolutions_backprojection=resolutions_backprojection,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm='batch_norm' in encoder_type,
                resolutions_depthwise_separable=resolutions_depthwise_separable_encoder)
        else:
            raise ValueError('Encoder type {} not supported.'.format(encoder_type))

        # Calculate number of channels for latent and skip connections combining image + depth
        latent_channels = n_filters_encoder[-1]
        n_skips = n_filters_encoder[:-1]
        n_skips = n_skips[::-1] + [0]

        # Build decoder
        if 'multiscale' in decoder_type:
            self.decoder = networks.MultiScaleDecoder(
                input_channels=latent_channels,
                output_channels=1,
                n_resolution=n_resolution_decoder,
                n_filters=n_filters_decoder,
                n_skips=n_skips,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                output_func='linear',
                use_batch_norm='batch_norm' in decoder_type,
                deconv_type='up')
        elif 'kbnet' in decoder_type:
            self.decoder = networks.KBNetDecoder(
                input_channels=n_filters_encoder[-1],
                output_channels=1,
                n_scale=1,
                n_filters=n_filters_decoder,
                n_skips=n_skips,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                output_func='linear',
                use_batch_norm='batch_norm' in decoder_type,
                deconv_type='up',
                resolutions_depthwise_separable=resolutions_depthwise_separable_decoder)
        else:
            raise ValueError('Decoder type {} not supported.'.format(decoder_type))

        # Move to device
        self.data_parallel()
        self.to(self.device)

    def forward(self,
                image,
                sparse_depth,
                validity_map,
                intrinsics):
        '''
        Forwards the inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W sparse depth
            validity_map : torch.Tensor[float32]
                N x 1 x H x W validity map of sparse depth
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix
        Returns:
            torch.Tensor[float32] : N x 1 x H x W output dense depth
        '''

        # Concatenate depth with validity map for N x 2 x H x W
        depth = torch.cat([sparse_depth, validity_map], dim=1)

        # Output is N x 8 x H x W
        depth = self.sparse_to_dense_pool(depth)

        latent, skips = self.encoder(image, depth, intrinsics)

        outputs = self.decoder(latent, skips=skips, shape=image.shape[-2:])

        output_depth = outputs[-1]
        output_depth = torch.sigmoid(output_depth)

        output_depth = \
            self.min_predict_depth / (output_depth + self.min_predict_depth / self.max_predict_depth)

        return output_depth

    def compute_loss(self,
                     output_depth0,
                     sparse_depth0,
                     validity_map0,
                     teacher_output0,
                     image0=None,
                     image1=None,
                     image2=None,
                     image3=None,
                     pose0to1=None,
                     pose0to2=None,
                     intrinsics0=None,
                     focal_length_baseline0=None,
                     w_stereo=1.00,
                     w_monocular=1.00,
                     w_color=0.15,
                     w_structure=0.95,
                     w_sparse_depth=0.00,
                     w_ensemble_depth=1.00,
                     w_ensemble_temperature=0.10,
                     w_sparse_select_ensemble=1.00,
                     sparse_select_dilate_kernel_size=3,
                     w_smoothness=0.00,
                     loss_func_ensemble_depth='smoothl1',
                     use_pose_for_ensemble=False,
                     ensemble_method='mondi'):
        '''
        Computes loss function

        Arg(s):
            output_depth0 : torch.Tensor[float32]
                N x 1 x H x W output depth for left image
            sparse_depth0 : torch.Tensor[float32]
                N x 1 x H x W sparse depth for left image
            validity_map0 : torch.Tensor[float32]
                N x 1 x H x W validity map of sparse depth for left image
            teacher_output0 : torch.Tensor[float32]
                N x M x H x W teacher output from ensemble for left image
            image0 : torch.Tensor[float32]
                N x 3 x H x W left image
            image1 : torch.Tensor[float32]
                N x 3 x H x W t-1 image
            image2 : torch.Tensor[float32]
                N x 3 x H x W t+1 image
            image3 : torch.Tensor[float32]
                N x 3 x H x W right image
            pose0to1 : torch.Tensor[float32]
                N x 4 x 4 transformation matrix
            pose0to2 : torch.Tensor[float32]
                N x 4 x 4 transformation matrix
            intrinsics0 : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix for left camera
            focal_length_baseline0 : torch.Tensor[float32]
                N x 2 focal length and baseline for left camera
            w_stereo : float
                weight of stereo reconstruction loss terms
            w_monocular : float
                weight of monocular reconstruction loss terms
            w_color : float
                weight of color consistency term
            w_structure : float
                weight of structure consistency term (SSIM)
            w_sparse_depth : float
                weight of sparse depth consistency term
            w_ensemble_depth : float
                weight of ensemble depth consistency term
            w_ensemble_temperature : float
                tempuerature for adaptive weighting of ensemble depth
            w_smoothness : float
                weight of smoothness term
            loss_func_ensemble_depth : str
                choose from l1, l2, smoothl1
            use_pose_for_ensemble : bool
                if True, then use video (rigid warping) for teacher criterion
            ensemble_method : str
                options: median, mean, random, mondi
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        shape = image0.shape

        # Keep track of number of views we have
        t = 0.0

        if image1 is not None:
            t = t + 1

        if image2 is not None:
            t = t + 1

        if image3 is not None:
            fb = focal_length_baseline0[:, 0] * focal_length_baseline0[:, 1]
            fb = fb[:, None, None, None]

            t = t + 1

        M = teacher_output0.shape[1]
        teacher_idxs = None

        ensemble_loss = loss_utils.structural_consistency_loss_func

        with torch.no_grad():
            if ensemble_method == 'mean':
                assert w_ensemble_temperature == 0
                teacher_output0 = torch.mean(teacher_output0, dim=1, keepdim=True)
            elif ensemble_method == 'median':
                assert w_ensemble_temperature == 0
                teacher_output0 = torch.median(teacher_output0, dim=1, keepdim=True)[0]
            elif ensemble_method == 'random':
                assert w_ensemble_temperature == 0
                r = random.randint(0, M-1)
                teacher_output0 = teacher_output0[:, r:r+1]
            elif ensemble_method == 'mondi':
                lossesxto0_ensemble = []

                if image3 is not None:
                    images3to0_ensemble = [
                        loss_utils.warp1d_horizontal(
                            image3,
                            -fb / teacher_output0[:, i:i+1])
                        for i in range(M)
                    ]
                    losses3to0_ensemble = [
                        ensemble_loss(image0, images3to0_ensemble[i], reduce_loss=False)
                        for i in range(M)
                    ]

                    lossesxto0_ensemble.append(losses3to0_ensemble)

                if pose0to2 is not None and pose0to1 is not None and use_pose_for_ensemble:
                    images1to0_ensemble = [
                        loss_utils.rigid_warp(
                            image1,
                            teacher_output0[:, i:i+1],
                            pose0to1,
                            intrinsics0,
                            shape)
                        for i in range(M)
                    ]
                    losses1to0_ensemble = [
                        ensemble_loss(image0, images1to0_ensemble[i], reduce_loss=False)
                        for i in range(M)
                    ]

                    lossesxto0_ensemble.append(losses1to0_ensemble)

                    images2to0_ensemble = [
                        loss_utils.rigid_warp(
                            image2,
                            teacher_output0[:, i:i+1],
                            pose0to2,
                            intrinsics0,
                            shape)
                        for i in range(M)
                    ]
                    losses2to0_ensemble = [
                        ensemble_loss(image0, images2to0_ensemble[i], reduce_loss=False)
                        for i in range(M)
                    ]

                    lossesxto0_ensemble.append(losses2to0_ensemble)

                # Take mean loss from each view
                losses_ensemble = []

                if len(lossesxto0_ensemble) != 0:
                    # Multiple with sparse depth error
                    sparse_select_ensemble_error = loss_utils.sparse_depth_error_weight(
                        sparse_depth=sparse_depth0,
                        dilation_kernel_size=sparse_select_dilate_kernel_size,
                        teacher_outputs=teacher_output0,
                        w_sparse_error=w_sparse_select_ensemble)

                    for i in range(M):
                        lossesxto0 = []

                        for lossxto0 in lossesxto0_ensemble:
                            lossesxto0.append(lossxto0[i])

                        # Minimum reprojection error
                        loss_ensemble, _ = torch.min(torch.cat(lossesxto0, dim=1), dim=1, keepdim=True)

                        losses_ensemble.append(loss_ensemble)

                    assert sparse_select_ensemble_error.shape[1] == len(losses_ensemble)

                    losses_ensemble = [
                        sparse_select_ensemble_error[:, i:i+1] * losses_ensemble[i]
                        for i in range(len(losses_ensemble))
                    ]

                # Aggregate ensemble into single teacher output
                teacher_output0, teacher_output_loss, teacher_idxs = \
                        loss_utils.aggregate_teacher_output(
                            losses_ensemble,
                            teacher_output0)

                DELTA = 0.3
                # Don't change teacher_output_loss
                non_error_sparse_map0 = torch.abs(teacher_output0 - sparse_depth0) < DELTA
                non_error_sparse_map0 = (validity_map0 * non_error_sparse_map0).bool()

                teacher_output0[non_error_sparse_map0] = sparse_depth0[non_error_sparse_map0]
            else:
                raise ValueError('No such ensemble method: {}'.format(ensemble_method))

        # Distillation loss
        w_adaptive_ensemble = torch.where(
            teacher_output0 > 0,
            torch.ones_like(teacher_output0),
            torch.zeros_like(teacher_output0))

        if w_ensemble_temperature > 0:
            w_adaptive_ensemble = w_adaptive_ensemble * torch.exp(-w_ensemble_temperature * teacher_output_loss)
            w_adaptive_unsupervised = 1 - w_adaptive_ensemble
        else:
            w_adaptive_unsupervised = torch.ones_like(w_adaptive_ensemble)

        # Unsupervised losses
        loss_color = []
        loss_structure = []

        if pose0to1 is not None or pose0to2 is not None:
            points = loss_utils.backproject_to_camera(output_depth0, intrinsics0, shape)

        if pose0to1 is not None:
            xy0to1 = loss_utils.project_to_pixel(points, pose0to1, intrinsics0, shape)
            image1to0 = loss_utils.grid_sample(image1, xy0to1, shape)

            loss_color1to0 = loss_utils.color_consistency_loss_func(
                image0,
                image1to0,
                reduce_loss=False)
            loss_structure1to0 = loss_utils.structural_consistency_loss_func(
                image0,
                image1to0,
                reduce_loss=False)

            loss_color.append(w_monocular * loss_color1to0)
            loss_structure.append(w_monocular * loss_structure1to0)
        else:
            image1to0 = None

        if pose0to2 is not None:
            xy0to2 = loss_utils.project_to_pixel(points, pose0to2, intrinsics0, shape)
            image2to0 = loss_utils.grid_sample(image2, xy0to2, shape)

            loss_color2to0 = loss_utils.color_consistency_loss_func(
                image0,
                image2to0,
                reduce_loss=False)

            loss_structure2to0 = loss_utils.structural_consistency_loss_func(
                image0,
                image2to0,
                reduce_loss=False)

            loss_color.append(w_monocular * loss_color2to0)
            loss_structure.append(w_monocular * loss_structure2to0)
        else:
            image2to0 = None

        if image3 is not None:
            image3to0 = loss_utils.warp1d_horizontal(image3, -fb / output_depth0)

            loss_color3to0 = loss_utils.color_consistency_loss_func(
                image0,
                image3to0,
                reduce_loss=False)

            loss_structure3to0 = loss_utils.structural_consistency_loss_func(
                image0,
                image3to0,
                reduce_loss=False)

            loss_color.append(w_stereo * loss_color3to0)
            loss_structure.append(w_stereo * loss_structure3to0)
        else:
            image3to0 = None

        # Take mean photometric reprojection across views
        loss_color = torch.mean(torch.stack(loss_color, dim=0), dim=0)
        loss_structure = torch.mean(torch.stack(loss_structure, dim=0), dim=0)

        loss_color = torch.mean(w_adaptive_unsupervised * loss_color)
        loss_structure = torch.mean(w_adaptive_unsupervised * loss_structure)

        if w_smoothness > 0:
            loss_smoothness = loss_utils.smoothness_loss_func(output_depth0, image0, w_adaptive_unsupervised)
        else:
            loss_smoothness = 0.0

        if w_sparse_depth > 0:
            loss_sparse_depth = loss_utils.sparse_depth_consistency_loss_func(
                output_depth0,
                sparse_depth0,
                w=validity_map0)
        else:
            loss_sparse_depth = 0.0

        if loss_func_ensemble_depth == 'l1':
            loss_ensemble_depth = loss_utils.l1_loss(
                output_depth0,
                teacher_output0,
                w=w_adaptive_ensemble)
        elif loss_func_ensemble_depth == 'l2':
            loss_ensemble_depth = loss_utils.l2_loss(
                output_depth0,
                teacher_output0,
                w=w_adaptive_ensemble)
        elif loss_func_ensemble_depth == 'smoothl1':
            loss_ensemble_depth = loss_utils.smooth_l1_loss(
                output_depth0,
                teacher_output0,
                w=w_adaptive_ensemble)
        elif loss_func_ensemble_depth == 'none' or w_ensemble_depth <= 0.0:
            loss_ensemble_depth = 0.0
        else:
            raise ValueError('No such loss: {}'.format(loss_func_ensemble_depth))

        loss = w_ensemble_depth * loss_ensemble_depth + \
            w_color * loss_color + \
            w_structure * loss_structure + \
            w_smoothness * loss_smoothness + \
            w_sparse_depth * loss_sparse_depth

        loss_info = {
            'loss_ensemble_depth': loss_ensemble_depth,
            'loss_color': loss_color,
            'loss_structure': loss_structure,
            'loss_smoothness': loss_smoothness,
            'loss_sparse_depth': loss_sparse_depth,
            'loss' : loss,
        }

        images_info = {
            'image1to0': image1to0,
            'image2to0': image2to0,
            'image3to0': image3to0,
            'teacher_output0': teacher_output0,
            'teacher_idxs': teacher_idxs,
        }

        return loss, loss_info, images_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list : list of parameters
        '''

        parameters = \
            list(self.sparse_to_dense_pool.parameters()) + \
            list(self.encoder.parameters()) + \
            list(self.decoder.parameters())

        return parameters

    def train(self):
        '''
        Sets model to training mode
        '''

        self.sparse_to_dense_pool.train()
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.sparse_to_dense_pool.eval()
        self.encoder.eval()
        self.decoder.eval()

    def to(self, device):
        '''
        Moves model to specified device

        Arg(s):
            device : torch.device
                device for running model
        '''

        # Move to device
        self.sparse_to_dense_pool.to(device)
        self.encoder.to(device)
        self.decoder.to(device)

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

        checkpoint = {}
        checkpoint['train_step'] = step
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Load weights for sparse_to_dense_depth, encoder, and decoder
        checkpoint['sparse_to_dense_pool_state_dict'] = self.sparse_to_dense_pool.state_dict()
        checkpoint['encoder_state_dict'] = self.encoder.state_dict()
        checkpoint['decoder_state_dict'] = self.decoder.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def restore_model(self, checkpoint_path, optimizer=None):
        '''
        Restore weights of the model

        Arg(s):
            checkpoint_path : str
                path to checkpoint
            optimizer : torch.optim
                optimizer
        Returns:
            int : current step in optimization
            torch.optim : optimizer with restored state
        '''

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore weights for sparse_to_dense_depth, encoder, and decoder
        self.sparse_to_dense_pool.load_state_dict(checkpoint['sparse_to_dense_pool_state_dict'])
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['train_step'], optimizer

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.sparse_to_dense_pool = torch.nn.DataParallel(self.sparse_to_dense_pool)
        self.encoder = torch.nn.DataParallel(self.encoder)
        self.decoder = torch.nn.DataParallel(self.decoder)

    def log_summary(self,
                    summary_writer,
                    tag,
                    step,
                    image0=None,
                    image1to0=None,
                    image2to0=None,
                    image3to0=None,
                    output_depth0=None,
                    sparse_depth0=None,
                    validity_map0=None,
                    teacher_output0=None,
                    ground_truth0=None,
                    pose0to1=None,
                    pose0to2=None,
                    scalars={},
                    n_display=4):
        '''
        Logs summary to Tensorboard

        Arg(s):
            summary_writer : SummaryWriter
                Tensorboard summary writer
            tag : str
                tag that prefixes names to log
            step : int
                current step in training
            image0 : torch.Tensor[float32]
                image from left camera
            image1to0 : torch.Tensor[float32]
                image at time step t-1 warped to time step t
            image2to0 : torch.Tensor[float32]
                image at time step t+1 warped to time step t
            image3to0 : torch.Tensor[float32]
                image from right camera reprojected to left camera
            output_depth0 : torch.Tensor[float32]
                output depth for left camera
            sparse_depth0 : torch.Tensor[float32]
                sparse_depth for left camera
            validity_map0 : torch.Tensor[float32]
                validity map for left camera
            teacher_output0 : torch.Tensor[float32]
                teacher output depth for left camera
            ground_truth0 : torch.Tensor[float32]
                ground truth depth for left camera
            pose0to1 : torch.Tensor[float32]
                N x 4 x 4 transformation matrix
            pose0to2 : torch.Tensor[float32]
                N x 4 x 4 transformation matrix
            scalars : dict[str, float]
                dictionary of scalars to log
            n_display : int
                number of images to display
        '''

        with torch.no_grad():

            display_summary_image = []
            display_summary_depth = []

            display_summary_image_text = tag
            display_summary_depth_text = tag

            if image0 is not None:
                image0_summary = image0[0:n_display, ...]

                display_summary_image_text += '_image0'
                display_summary_depth_text += '_image0'

                # Add to list of images to log
                display_summary_image.append(
                    torch.cat([
                        image0_summary.cpu(),
                        torch.zeros_like(image0_summary, device=torch.device('cpu'))],
                        dim=-1))

                display_summary_depth.append(display_summary_image[-1])

            image_list = []
            image_text_list = []

            if image1to0 is not None:
                image_list.append(image1to0)
                image_text_list.append('_image1to0-error')

            if image2to0 is not None:
                image_list.append(image2to0)
                image_text_list.append('_image2to0-error')

            if image3to0 is not None:
                image_list.append(image3to0)
                image_text_list.append('_image3to0-error')

            for image_disp, image_disp_txt in zip(image_list, image_text_list):

                if image0 is not None and image_disp is not None:
                    image_disp_summary = image_disp[0:n_display, ...]

                    display_summary_image_text += image_disp_txt

                    # Compute reconstruction error w.r.t. image 0
                    image_disp_error_summary = torch.mean(
                        torch.abs(image0_summary - image_disp_summary),
                        dim=1,
                        keepdim=True)

                    # Add to list of images to log
                    image_disp_error_summary = log_utils.colorize(
                        (image_disp_error_summary / 0.10).cpu(),
                        colormap='inferno')

                    display_summary_image.append(
                        torch.cat([
                            image_disp_summary.cpu(),
                            image_disp_error_summary],
                            dim=3))

            if output_depth0 is not None:
                output_depth0_summary = output_depth0[0:n_display, ...]

                display_summary_depth_text += '_output0'

                # Add to list of images to log
                n_batch, _, n_height, n_width = output_depth0_summary.shape

                display_summary_depth.append(
                    torch.cat([
                        log_utils.colorize(
                            (output_depth0_summary / self.max_predict_depth).cpu(),
                            colormap='viridis'),
                        torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
                        dim=3))

                # Log distribution of output depth
                summary_writer.add_histogram(tag + '_output_depth0_distro', output_depth0, global_step=step)

            if output_depth0 is not None and sparse_depth0 is not None and validity_map0 is not None:
                sparse_depth0_summary = sparse_depth0[0:n_display]
                validity_map0_summary = validity_map0[0:n_display]

                display_summary_depth_text += '_sparse0-error'

                # Compute output error w.r.t. input sparse depth
                sparse_depth0_error_summary = \
                    torch.abs(output_depth0_summary - sparse_depth0_summary)

                sparse_depth0_error_summary = torch.where(
                    validity_map0_summary == 1.0,
                    sparse_depth0_error_summary / (sparse_depth0_summary + 1e-8),
                    validity_map0_summary)

                # Add to list of images to log
                sparse_depth0_summary = log_utils.colorize(
                    (sparse_depth0_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                sparse_depth0_error_summary = log_utils.colorize(
                    (sparse_depth0_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        sparse_depth0_summary,
                        sparse_depth0_error_summary],
                        dim=3))

                # Log distribution of sparse depth
                summary_writer.add_histogram(tag + '_sparse_depth0_distro', sparse_depth0, global_step=step)

            if output_depth0 is not None and teacher_output0 is not None:
                teacher_output0_summary = teacher_output0[0:n_display]

                display_summary_depth_text += '_teacheroutput0-error'

                # Compute output error w.r.t. teacher output
                teacher_output0_error_summary = \
                    torch.abs(output_depth0_summary - teacher_output0_summary)

                teacher_output0_error_summary = torch.where(
                    teacher_output0_summary > 0,
                    teacher_output0_error_summary / (teacher_output0_summary + 1e-8),
                    torch.zeros_like(teacher_output0_summary))

                # Add to list of images to log
                teacher_output0_summary = log_utils.colorize(
                    (teacher_output0_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                teacher_output0_error_summary = log_utils.colorize(
                    (teacher_output0_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        teacher_output0_summary,
                        teacher_output0_error_summary],
                        dim=3))

                # Log distribution of teacher output
                summary_writer.add_histogram(tag + '_teacher_output0_distro', teacher_output0, global_step=step)

            if output_depth0 is not None and ground_truth0 is not None:
                validity_map0 = torch.unsqueeze(ground_truth0[:, 1, :, :], dim=1)
                ground_truth0 = torch.unsqueeze(ground_truth0[:, 0, :, :], dim=1)

                validity_map0_summary = validity_map0[0:n_display]
                ground_truth0_summary = ground_truth0[0:n_display]

                display_summary_depth_text += '_groundtruth0-error'

                # Compute output error w.r.t. ground truth
                ground_truth0_error_summary = \
                    torch.abs(output_depth0_summary - ground_truth0_summary)

                ground_truth0_error_summary = torch.where(
                    validity_map0_summary == 1.0,
                    (ground_truth0_error_summary + 1e-8) / (ground_truth0_summary + 1e-8),
                    validity_map0_summary)

                # Add to list of images to log
                ground_truth0_summary = log_utils.colorize(
                    (ground_truth0_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                ground_truth0_error_summary = log_utils.colorize(
                    (ground_truth0_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        ground_truth0_summary,
                        ground_truth0_error_summary],
                        dim=3))

                # Log distribution of ground truth
                summary_writer.add_histogram(tag + '_ground_truth0_distro', ground_truth0, global_step=step)

            if pose0to1 is not None:
                # Log distribution of pose 1 to 0translation vector
                summary_writer.add_histogram(tag + '_tx01_distro', pose0to1[:, 0, 3], global_step=step)
                summary_writer.add_histogram(tag + '_ty01_distro', pose0to1[:, 1, 3], global_step=step)
                summary_writer.add_histogram(tag + '_tz01_distro', pose0to1[:, 2, 3], global_step=step)

            if pose0to2 is not None:
                # Log distribution of pose 2 to 0 translation vector
                summary_writer.add_histogram(tag + '_tx02_distro', pose0to2[:, 0, 3], global_step=step)
                summary_writer.add_histogram(tag + '_ty02_distro', pose0to2[:, 1, 3], global_step=step)
                summary_writer.add_histogram(tag + '_tz02_distro', pose0to2[:, 2, 3], global_step=step)

            # Log scalars to tensorboard
            for (name, value) in scalars.items():
                summary_writer.add_scalar(tag + '_' + name, value, global_step=step)

            # Log image summaries to tensorboard
            if len(display_summary_image) > 1:
                display_summary_image = torch.cat(display_summary_image, dim=2)

                summary_writer.add_image(
                    display_summary_image_text,
                    torchvision.utils.make_grid(display_summary_image, nrow=n_display),
                    global_step=step)

            if len(display_summary_depth) > 1:
                display_summary_depth = torch.cat(display_summary_depth, dim=2)

                summary_writer.add_image(
                    display_summary_depth_text,
                    torchvision.utils.make_grid(display_summary_depth, nrow=n_display),
                    global_step=step)
