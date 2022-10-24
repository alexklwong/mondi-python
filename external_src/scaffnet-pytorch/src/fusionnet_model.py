import torch, torchvision
import log_utils, losses, loss_utils, networks, net_utils


class FusionNetModel(object):
    '''
    Network to fuse image and depth information together and learns residual for initial depth

    Arg(s):
        encoder_type : list[str]
            encoder types: vggnet08, vggnet11, vggnet13, batch_norm
        n_filters_encoder_image : list[int]
            number of filters to use in each block of image encoder
        n_filters_encoder_depth : list[int]
            number of filters to use in each block of depth encoder
        decoder_type : list[str]
            decoder_types : multi-scale, batch_norm
        n_filters_decoder : list[int]
            number of filters to use in each block of decoder
        scale_match_method : str
            scale matching method: replace, local_scale, none
        scale_match_kernel_size : int
            kernel size for scaling or replacing
        min_multiplier_depth : float
            minimum depth multiplier supported by model
        max_multiplier_depth : float
            maximum depth multiplier supported by model
        min_residual_depth : float
            minimum depth residual supported by model
        max_residual_depth : float
            maximum depth residual supported by model
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function for network
        device : torch.device
            device for running model
    '''

    def __init__(self,
                 encoder_type,
                 n_filters_encoder_image,
                 n_filters_encoder_depth,
                 decoder_type,
                 n_filters_decoder,
                 # Input and output settings
                 scale_match_method,
                 scale_match_kernel_size,
                 min_predict_depth,
                 max_predict_depth,
                 min_multiplier_depth,
                 max_multiplier_depth,
                 min_residual_depth,
                 max_residual_depth,
                 # Weight settings
                 weight_initializer,
                 activation_func,
                 device=torch.device('cuda')):

        self.scale_match_method = scale_match_method
        self.scale_match_kernel_size = scale_match_kernel_size
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.min_multiplier_depth = min_multiplier_depth
        self.max_multiplier_depth = max_multiplier_depth
        self.min_residual_depth = min_residual_depth
        self.max_residual_depth = max_residual_depth
        self.device = device

        n_filters_encoder = [
            i + z
            for i, z in zip(n_filters_encoder_image, n_filters_encoder_depth)
        ]
        n_skips = n_filters_encoder[:-1]
        n_skips = n_skips[::-1] + [0]

        # Build network
        if 'vggnet08' in encoder_type:
            self.encoder_image = networks.VGGNetEncoder(
                n_layer=8,
                input_channels=3,
                n_filters=n_filters_encoder_image,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm='batch_norm' in encoder_type)
            self.encoder_depth = networks.VGGNetEncoder(
                n_layer=8,
                input_channels=2,
                n_filters=n_filters_encoder_depth,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm='batch_norm' in encoder_type)
        elif 'vggnet11' in encoder_type:
            self.encoder_image = networks.VGGNetEncoder(
                n_layer=11,
                input_channels=3,
                n_filters=n_filters_encoder_image,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm='batch_norm' in encoder_type)
            self.encoder_depth = networks.VGGNetEncoder(
                n_layer=11,
                input_channels=2,
                n_filters=n_filters_encoder_depth,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm='batch_norm' in encoder_type)

        if 'multi-scale' in decoder_type:
            self.decoder = networks.MultiScaleDecoder(
                input_channels=n_filters_encoder[-1],
                output_channels=2,
                n_resolution=1,
                n_filters=n_filters_decoder,
                n_skips=n_skips,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                output_func='linear',
                use_batch_norm='batch_norm' in decoder_type,
                deconv_type='up')

        # Move to device
        self.data_parallel()
        self.to(self.device)

    def forward(self, image, input_depth, sparse_depth):
        '''
        Forwards the inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            input_depth : torch.Tensor[float32]
                N x 1 x H x W input dense depth
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W sparse depth
        Returns:
            torch.Tensor[float32] : output dense depth
        '''

        # Scale the input points based on the filtered sparse depth
        input_depth = net_utils.scale_match(
            input_depth=input_depth,
            sparse_depth=sparse_depth,
            method=self.scale_match_method,
            kernel_size=self.scale_match_kernel_size)

        # Inputs to network can be include
        # (1) predicted depth, (2) raw sparse depth
        depth = torch.cat([input_depth, sparse_depth], dim=1)

        # Forward through the network
        latent_image, skips_image = self.encoder_image(image)
        latent_depth, skips_depth = self.encoder_depth(depth)

        latent = torch.cat([latent_image, latent_depth], dim=1)

        skips = [
            torch.cat([skip_image, skip_depth], dim=1)
            for skip_image, skip_depth in zip(skips_image, skips_depth)
        ]

        output = self.decoder(
            x=latent,
            skips=skips,
            shape=sparse_depth.shape[-2:])[-1]

        # Construct multiplicative and additive residual output
        output_multiplier_depth = torch.unsqueeze(output[:, 0, :, :], dim=1)
        output_multiplier_depth = torch.sigmoid(output_multiplier_depth)

        # y = (max - min) * x + min
        range_multiplier_depth = self.max_multiplier_depth - self.min_multiplier_depth
        output_multiplier_depth = \
            range_multiplier_depth * output_multiplier_depth + self.min_multiplier_depth

        output_residual_depth = torch.unsqueeze(output[:, 1, :, :], dim=1)

        output_residual_depth = torch.clamp(
            output_residual_depth,
            min=self.min_residual_depth,
            max=self.max_residual_depth)

        # y = alpha * x + beta
        output_depth = output_multiplier_depth * input_depth + output_residual_depth

        output_depth = torch.clamp(
            output_depth,
            min=self.min_predict_depth,
            max=self.max_predict_depth)

        return output_depth

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

        shape = image0.shape

        # Backproject points to 3D camera coordinates
        points = loss_utils.backproject_to_camera(output_depth0, intrinsics, shape)

        # Reproject points onto image 1 and image 2
        target_xy0to1 = loss_utils.project_to_pixel(points, pose0to1, intrinsics, shape)
        target_xy0to2 = loss_utils.project_to_pixel(points, pose0to2, intrinsics, shape)

        # Reconstruct image0 from image1 and image2 by reprojection
        image1to0 = loss_utils.grid_sample(image1, target_xy0to1, shape)
        image2to0 = loss_utils.grid_sample(image2, target_xy0to2, shape)

        # Color consistency loss function
        loss_color1to0 = losses.color_consistency_loss_func(
            src=image1to0,
            tgt=image0)

        loss_color2to0 = losses.color_consistency_loss_func(
            src=image2to0,
            tgt=image0)

        loss_color = loss_color1to0 + loss_color2to0

        # Structural consistency loss function
        loss_structure1to0 = losses.structural_consistency_loss_func(
            src=image1to0,
            tgt=image0)

        loss_structure2to0 = losses.structural_consistency_loss_func(
            src=image2to0,
            tgt=image0)

        loss_structure = loss_structure1to0 + loss_structure2to0

        # Sparse depth consistency loss function
        loss_sparse_depth = losses.sparse_depth_consistency_loss_func(
            src=output_depth0,
            tgt=sparse_depth0,
            w=validity_map0)

        # Local smoothness loss function
        loss_smoothness = losses.smoothness_loss_func(
            predict=output_depth0,
            image=image0)

        # Prior depth consistency loss function
        loss_prior_depth = 0.0

        if w_prior_depth > 0.0 and loss_color1to0 < threshold_prior_depth:
            # Reproject points onto image 1
            points_prior = net_utils.backproject_to_camera(input_depth0, intrinsics, shape)
            target_xy0to1_prior = net_utils.project_to_pixel(points_prior, pose0to1, intrinsics, shape)

            # Reconstruct image0 from image1 and image2 by reprojection
            image1to0_prior = net_utils.grid_sample(image1, target_xy0to1_prior, shape)

            image1to0_prior_error = torch.sum(torch.abs(image1to0_prior - image0), dim=1)
            image1to0_error = torch.sum(torch.abs(image1to0 - image0), dim=1)

            validity_map0_prior = torch.where(
                image1to0_prior_error <= image1to0_error,
                torch.ones_like(output_depth0),
                torch.zeros_like(output_depth0))

            loss_prior_depth = losses.prior_depth_consistency_loss_func(
                src=output_depth0,
                tgt=input_depth0,
                w=validity_map0_prior)

        # l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm} + w_{tp}l_{tp}
        loss = \
            w_color * loss_color + \
            w_structure * loss_structure + \
            w_sparse_depth * loss_sparse_depth + \
            w_smoothness * loss_smoothness + \
            w_prior_depth * loss_prior_depth

        loss_info = {
            'loss_color' : loss_color,
            'loss_structure' : loss_structure,
            'loss_sparse_depth' : loss_sparse_depth,
            'loss_smoothness' : loss_smoothness,
            'loss_prior_depth' : loss_prior_depth,
            'loss' : loss,
            'image1to0' : image1to0,
            'image2to0' : image2to0
        }

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list : list of parameters
        '''

        return list(self.encoder_image.parameters()) + \
            list(self.encoder_depth.parameters()) + \
            list(self.decoder.parameters())

    def train(self):
        '''
        Sets model to training mode
        '''

        self.encoder_image.train()
        self.encoder_depth.train()
        self.decoder.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.encoder_image.eval()
        self.encoder_depth.eval()
        self.decoder.eval()

    def to(self, device):
        '''
        Moves model to specified device

        Arg(s):
            device : torch.device
                device for running model
        '''

        # Move to device
        self.encoder_image.to(device)
        self.encoder_depth.to(device)
        self.decoder.to(device)

    def save_model(self, checkpoint_path, step, optimizer, scaffnet_model=None):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
            scaffnet_model : ScaffNetModel
                instance of ScaffNet
        '''

        checkpoint = {}

        if scaffnet_model is not None:
            # Save encoder and decoder weights
            if 'spatial_pyramid_pool' in scaffnet_model.encoder_type:
                checkpoint['spatial_pyramid_pool_state_dict'] = scaffnet_model.spatial_pyramid_pool.state_dict()

            if 'uncertainty' in scaffnet_model.decoder_type:
                checkpoint['scaffnet_decoder_uncertainty_state_dict'] = scaffnet_model.decoder_uncertainty.state_dict()

            checkpoint['scaffnet_encoder_state_dict'] = scaffnet_model.encoder.state_dict()
            checkpoint['scaffnet_decoder_depth_state_dict'] = scaffnet_model.decoder_depth.state_dict()

        # Save training state
        checkpoint['train_step'] = step
        checkpoint['fusionnet_optimizer_state_dict'] = optimizer.state_dict()

        # Save encoder and decoder weights
        checkpoint['fusionnet_encoder_image_state_dict'] = self.encoder_image.state_dict()
        checkpoint['fusionnet_encoder_depth_state_dict'] = self.encoder_depth.state_dict()
        checkpoint['fusionnet_decoder_state_dict'] = self.decoder.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def restore_model(self, checkpoint_path, optimizer=None, scaffnet_model=None):
        '''
        Restore weights of the model

        Arg(s):
            checkpoint_path : str
                path to checkpoint
            optimizer : torch.optim
                optimizer
            scaffnet_model : ScaffNetModel
                instance of ScaffNet
        Returns:
            int : current step in optimization
            torch.optim : optimizer with restored state
        '''

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if scaffnet_model is not None:
            # Save encoder and decoder weights
            if 'spatial_pyramid_pool' in scaffnet_model.encoder_type:
                scaffnet_model.spatial_pyramid_pool.load_state_dict(checkpoint['spatial_pyramid_pool_state_dict'])

            if 'uncertainty' in scaffnet_model.decoder_type:
                scaffnet_model.decoder_uncertainty.load_state_dict(checkpoint['scaffnet_decoder_uncertainty_state_dict'])

            scaffnet_model.encoder.load_state_dict(checkpoint['scaffnet_encoder_state_dict'])
            scaffnet_model.decoder_depth.load_state_dict(checkpoint['scaffnet_decoder_depth_state_dict'])

        # Restore encoder and decoder weights
        self.encoder_image.load_state_dict(checkpoint['fusionnet_encoder_image_state_dict'])
        self.encoder_depth.load_state_dict(checkpoint['fusionnet_encoder_depth_state_dict'])

        self.decoder.load_state_dict(checkpoint['fusionnet_decoder_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['fusionnet_optimizer_state_dict'])

        # Return the current step and optimizer
        return checkpoint['train_step'], optimizer

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.encoder_image = torch.nn.DataParallel(self.encoder_image)
        self.encoder_depth = torch.nn.DataParallel(self.encoder_depth)
        self.decoder = torch.nn.DataParallel(self.decoder)

    def log_summary(self,
                    summary_writer,
                    tag,
                    step,
                    image0=None,
                    image1to0=None,
                    image2to0=None,
                    output_depth0=None,
                    sparse_depth0=None,
                    validity_map0=None,
                    input_depth0=None,
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
                image at time step t
            image1to0 : torch.Tensor[float32]
                image at time step t-1 warped to time step t
            image2to0 : torch.Tensor[float32]
                image at time step t+1 warped to time step t
            output_depth0 : torch.Tensor[float32]
                output depth for time step t
            sparse_depth0 : torch.Tensor[float32]
                sparse_depth for time step t
            validity_map0 : torch.Tensor[float32]
                validity map for time step t
            input_depth0 : torch.Tensor[float32]
                input depth for time step t
            ground_truth0 : torch.Tensor[float32]
                ground truth depth for time step t
            pose0to1 : torch.Tensor[float32]
                N x 4 x 4 transformation matrix from time t to t-1
            pose0to2 : torch.Tensor[float32]
                N x 4 x 4 transformation matrix from time t to t+1
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

            if output_depth0 is not None and input_depth0 is not None:
                input_depth0_summary = input_depth0[0:n_display]

                display_summary_depth_text += '_input0-error'

                # Compute output error w.r.t. pseudo ground truth
                input_depth0_error_summary = \
                    torch.abs(output_depth0_summary - input_depth0_summary)

                input_depth0_error_summary = torch.where(
                    input_depth0_summary > 0,
                    input_depth0_error_summary / (input_depth0_summary + 1e-8),
                    torch.zeros_like(input_depth0_summary))

                # Add to list of images to log
                input_depth0_summary = log_utils.colorize(
                    (input_depth0_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                input_depth0_error_summary = log_utils.colorize(
                    (input_depth0_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        input_depth0_summary,
                        input_depth0_error_summary],
                        dim=3))

                # Log distribution of pseudo ground truth
                summary_writer.add_histogram(tag + '_input_depth0_distro', input_depth0, global_step=step)

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
