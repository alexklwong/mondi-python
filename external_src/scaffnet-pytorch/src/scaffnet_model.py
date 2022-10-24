import torch, torchvision
import log_utils, losses, networks


class ScaffNetModel(object):
    '''
    ScaffNet to infer topology from sparse points

    Arg(s):
        encoder_type : list[str]
            encoder types: vggnet08, vggnet11, vggnet13, spatial_pyramid_pool
        n_filters_encoder : list[int]
            number of filters to use for each block in encoder
        decoder_type : list[str]
            decoder types: multi-scale
        n_filters_decoder : list[int]
            number of filters to use for each block in decoder
        max_pool_sizes_spatial_pyramid_pool : list[int]
            list of max pooling sizes for spatial pyramid pooling
        n_convolution_spatial_pyramid_pool : int
            number of 1 x 1 convolutions to use for spatial pyramid pooling
        n_filter_spatial_pyramid_pool : int
            number of filters to use for 1 x 1 convolutions
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function for network
        min_predict_depth : float
            minimum depth prediction supported by model
        max_predict_depth : float
            maximum depth prediction supported by model
        device : torch.device
            device for running model
    '''

    def __init__(self,
                 encoder_type,
                 n_filters_encoder,
                 decoder_type,
                 n_filters_decoder,
                 max_pool_sizes_spatial_pyramid_pool,
                 n_convolution_spatial_pyramid_pool,
                 n_filter_spatial_pyramid_pool,
                 weight_initializer,
                 activation_func,
                 min_predict_depth,
                 max_predict_depth,
                 device=torch.device('cuda')):

        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.device = device

        # Build network
        input_channels = 1
        n_skips = n_filters_encoder[0:-1][::-1] + [0]

        if 'spatial_pyramid_pool' in encoder_type:
            input_channels += input_channels * len(max_pool_sizes_spatial_pyramid_pool)

            self.spatial_pyramid_pool = networks.SpatialPyramidPool(
                input_channels=input_channels,
                pool_sizes=max_pool_sizes_spatial_pyramid_pool,
                n_filter=n_filter_spatial_pyramid_pool,
                n_convolution=n_convolution_spatial_pyramid_pool,
                weight_initializer=weight_initializer,
                activation_func=activation_func)

            # Adjust input channels and skip connections for encoder
            input_channels = n_filter_spatial_pyramid_pool
            n_skips = n_filters_encoder[0:-1][::-1] + [n_filter_spatial_pyramid_pool]

        if 'vggnet08' in encoder_type:
            self.encoder = networks.VGGNetEncoder(
                n_layer=8,
                input_channels=input_channels,
                n_filters=n_filters_encoder,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm='batch_norm' in encoder_type,
                use_instance_norm='instance_norm' in encoder_type)

        elif 'resnet18' in encoder_type:
            self.encoder = networks.ResNetEncoder(
                n_layer=18,
                input_channels=input_channels,
                n_filters=n_filters_encoder,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm='batch_norm' in encoder_type,
                use_instance_norm='instance_norm' in encoder_type)

        if 'multi-scale' in decoder_type:
            self.decoder_depth = networks.MultiScaleDecoder(
                input_channels=n_filters_encoder[-1],
                output_channels=1,
                n_resolution=1,
                n_filters=n_filters_decoder,
                n_skips=n_skips,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                output_func='sigmoid',
                use_batch_norm='batch_norm' in decoder_type,
                use_instance_norm='instance_norm' in decoder_type,
                deconv_type='up')

        if 'uncertainty' in decoder_type:
            self.decoder_uncertainty = networks.MultiScaleDecoder(
                input_channels=n_filters_encoder[-1],
                output_channels=1,
                n_resolution=1,
                n_filters=n_filters_decoder,
                n_skips=n_skips,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                output_func='linear',
                use_batch_norm='batch_norm' in decoder_type,
                use_instance_norm='instance_norm' in decoder_type,
                deconv_type='up')

        # Move to device
        self.to(self.device)

    def forward(self, sparse_depth):
        '''
        Forwards the inputs through the network

        Arg(s):
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W sparse depth map
        Returns:
            torch.Tensor[float32] : N x 1 x H x W dense depth
        '''

        sparse_depth = torch.where(
            sparse_depth > self.max_predict_depth,
            self.max_predict_depth * torch.ones_like(sparse_depth),
            sparse_depth)

        encoder_input = sparse_depth

        # Forward through the network
        if 'spatial_pyramid_pool' in self.encoder_type:
            encoder_input = self.spatial_pyramid_pool(encoder_input)

        latent, skips = self.encoder(encoder_input)

        if 'spatial_pyramid_pool' in self.encoder_type:
            skips = [encoder_input] + skips

        output_depth = self.decoder_depth(
            x=latent,
            skips=skips,
            shape=sparse_depth.shape[-2:])[-1]

        # Convert inverse depth to depth
        output_depth = \
            self.min_predict_depth / (output_depth + self.min_predict_depth / self.max_predict_depth)

        if 'uncertainty' in self.decoder_type:
            output_uncertainty = self.decoder_uncertainty(latent, skips)[-1]
            output_depth = torch.cat([output_depth, output_uncertainty], dim=1)

        return output_depth

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

        validity_map_target_depth = torch.where(
            target_depth > 0.0,
            torch.ones_like(target_depth),
            torch.zeros_like(target_depth))

        if 'uncertainty' in self.decoder_type and output_uncertainty is not None:
            loss_supervised = losses.l1_with_uncertainty_loss_func(
                src=output_depth,
                tgt=target_depth,
                uncertainty=output_uncertainty,
                w=validity_map_target_depth)
        else:
            if 'supervised_l1' in loss_func:
                loss_supervised = losses.l1_loss_func(
                    src=output_depth,
                    tgt=target_depth,
                    w=validity_map_target_depth)

            elif 'supervised_l1_normalized' in loss_func:
                loss_supervised = losses.l1_loss_func(
                    src=output_depth,
                    tgt=target_depth,
                    w=validity_map_target_depth,
                    normalize=True)

        loss = w_supervised * loss_supervised

        loss_info = {
            'loss' : loss,
        }

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list : list of parameters
        '''

        if 'spatial_pyramid_pool' in self.encoder_type:
            parameters = \
                list(self.spatial_pyramid_pool.parameters()) + \
                list(self.encoder.parameters()) + \
                list(self.decoder_depth.parameters())
        else:
            parameters = \
                list(self.encoder.parameters()) + \
                list(self.decoder_depth.parameters())

        if 'uncertainty' in self.decoder_type:
            parameters = parameters + list(self.decoder_uncertainty.parameters())

        return parameters

    def train(self):
        '''
        Sets model to training mode
        '''

        if 'spatial_pyramid_pool' in self.encoder_type:
            self.spatial_pyramid_pool.train()

        if 'uncertainty' in self.decoder_type:
            self.decoder_uncertainty.train()

        self.encoder.train()
        self.decoder_depth.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        if 'spatial_pyramid_pool' in self.encoder_type:
            self.spatial_pyramid_pool.eval()

        if 'uncertainty' in self.decoder_type:
            self.decoder_uncertainty.eval()

        self.encoder.eval()
        self.decoder_depth.eval()

    def to(self, device):
        '''
        Moves model to specified device

        Arg(s):
            device : torch.device
                device for running model
        '''

        if 'spatial_pyramid_pool' in self.encoder_type:
            self.spatial_pyramid_pool.to(device)

        if 'uncertainty' in self.decoder_type:
            self.decoder_uncertainty.to(device)

        self.encoder.to(device)
        self.decoder_depth.to(device)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        if 'spatial_pyramid_pool' in self.encoder_type:
            self.spatial_pyramid_pool = torch.nn.DataParallel(self.spatial_pyramid_pool)

        if 'uncertainty' in self.decoder_type:
            self.decoder_uncertainty = torch.nn.DataParallel(self.decoder_uncertainty)

        self.encoder = torch.nn.DataParallel(self.encoder)
        self.decoder_depth = torch.nn.DataParallel(self.decoder_depth)

    def freeze(self, module_names=[]):
        '''
        Freeze a portion of the network

        Arg(s):
            module_names : list[str]
                spatial_pyramid_pool, encoder, decoder_depth, decoder_uncertainty, all
        '''

        if module_names is None or 'none' in module_names:
            module_names = []
        # Add all modules to the list of modules
        elif 'all' in module_names:
            module_names = [
                'spatial_pyramid_pool',
                'encoder',
                'decoder_depth',
                'decoder_uncertainty'
            ]

        # Freeze modules
        for module_name in module_names:

            try:
                module = getattr(self, module_name)

                for param in module.parameters():
                    param.requires_grad = False

                module.eval()

            except AttributeError:
                exception_text = 'Unsupported module name: {}\n'.format(module_name)
                exception_text += 'Must be one of spatial_pyramid_pool, encoder, decoder_depth, decoder_uncertainty, all'
                raise ValueError(exception_text)

    def unfreeze(self):
        '''
        Unfreeze the network
        '''

        module_names = [
            'spatial_pyramid_pool',
            'encoder',
            'decoder_depth',
            'decoder_uncertainty'
        ]

        # Unfreeze modules
        for module_name in module_names:

            try:
                module = getattr(self, module_name)

                for param in module.parameters():
                    param.requires_grad = True

            except AttributeError:
                # If this happen it is because we didn't find a module, so pass
                pass

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
        # Save training state
        checkpoint['train_step'] = step
        checkpoint['scaffnet_optimizer_state_dict'] = optimizer.state_dict()

        # Save encoder and decoder weights
        if 'spatial_pyramid_pool' in self.encoder_type:
            checkpoint['spatial_pyramid_pool_state_dict'] = self.spatial_pyramid_pool.state_dict()

        if 'uncertainty' in self.decoder_type:
            checkpoint['scaffnet_decoder_uncertainty_state_dict'] = self.decoder_uncertainty.state_dict()

        checkpoint['scaffnet_encoder_state_dict'] = self.encoder.state_dict()
        checkpoint['scaffnet_decoder_depth_state_dict'] = self.decoder_depth.state_dict()

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
        keys = checkpoint.keys()

        do_restore_optimizer = optimizer is not None

        try:
            # Restore encoder and decoder weights
            self.encoder.load_state_dict(checkpoint['scaffnet_encoder_state_dict'])
            self.decoder_depth.load_state_dict(checkpoint['scaffnet_decoder_depth_state_dict'])

            if 'spatial_pyramid_pool' in self.encoder_type:
                if 'spatial_pyramid_pool_state_dict' in keys:
                    self.spatial_pyramid_pool.load_state_dict(checkpoint['spatial_pyramid_pool_state_dict'])
                else:
                    do_restore_optimizer = False

            if 'uncertainty' in self.decoder_type:
                if 'scaffnet_decoder_uncertainty_state_dict' in keys:
                    self.decoder_uncertainty.load_state_dict(checkpoint['scaffnet_decoder_uncertainty_state_dict'])
                else:
                    do_restore_optimizer = False

            if do_restore_optimizer:
                optimizer.load_state_dict(checkpoint['scaffnet_optimizer_state_dict'])
        except Exception as e:
            print('Encountered: {}'.format(e))
            print('Restoring using keys without scaffnet prefix')
            # Restore encoder and decoder weights
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.decoder_depth.load_state_dict(checkpoint['decoder_depth_state_dict'])

            if 'spatial_pyramid_pool' in self.encoder_type:
                if 'spatial_pyramid_pool' in keys:
                    self.spatial_pyramid_pool.load_state_dict(checkpoint['spatial_pyramid_pool_state_dict'])
                else:
                    do_restore_optimizer = False

            if 'uncertainty' in self.decoder_type:
                if 'uncertainty' in keys:
                    self.decoder_uncertainty.load_state_dict(checkpoint['decoder_uncertainty_state_dict'])
                else:
                    do_restore_optimizer = False

            if do_restore_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Return the current step and optimizer
        return checkpoint['train_step'], optimizer

    def log_summary(self,
                    summary_writer,
                    tag,
                    step,
                    output_depth=None,
                    output_uncertainty=None,
                    ground_truth=None,
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
            output_depth : torch.Tensor[float32]
                N x 1 x H x W output depth
            output_uncertainty : torch.Tensor[float32]
                N x 1 x H x W output uncertainty
            ground_truth : torch.Tensor[float32]
                N x 2 x H x W ground truth depth
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

            if output_depth is not None:
                output_depth_summary = output_depth[0:n_display, ...]

                display_summary_depth_text += '_output'

                # Add to list of images to log
                n_batch, _, n_height, n_width = output_depth_summary.shape

                if output_uncertainty is not None:
                    output_uncertainty_summary = output_uncertainty[0:n_display, ...]

                    output_uncertainty_summary = 4.0 * torch.exp(output_uncertainty_summary)

                    # Log distribution of output depth
                    summary_writer.add_histogram(tag + '_output_uncertainty_distro', output_uncertainty, global_step=step)
                else:
                    output_uncertainty_summary = \
                        torch.zeros(n_batch, 1, n_height, n_width, device=torch.device('cpu'))

                display_summary_depth.append(
                    torch.cat([
                        log_utils.colorize(
                            (output_depth_summary / self.max_predict_depth).cpu(),
                            colormap='viridis'),
                        log_utils.colorize(
                            (output_uncertainty_summary / self.max_predict_depth).cpu(),
                            colormap='inferno')],
                        dim=3))

                # Log distribution of output depth
                summary_writer.add_histogram(tag + '_output_depth_distro', output_depth, global_step=step)

            if output_depth is not None and ground_truth is not None:
                validity_map = torch.unsqueeze(ground_truth[:, 1, :, :], dim=1)
                ground_truth = torch.unsqueeze(ground_truth[:, 0, :, :], dim=1)

                validity_map_summary = validity_map[0:n_display, ...]
                ground_truth_summary = ground_truth[0:n_display, ...]

                display_summary_depth_text += '_groundtruth-error'

                # Compute output error w.r.t. ground truth
                ground_truth_error_summary = \
                    torch.abs(output_depth_summary - ground_truth_summary)

                ground_truth_error_summary = torch.where(
                    validity_map_summary == 1.0,
                    (ground_truth_error_summary + 1e-8) / (ground_truth_summary + 1e-8),
                    validity_map_summary)

                # Add to list of images to log
                ground_truth_summary = log_utils.colorize(
                    (ground_truth_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                ground_truth_error_summary = log_utils.colorize(
                    (ground_truth_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        ground_truth_summary,
                        ground_truth_error_summary],
                        dim=3))

                # Log distribution of ground truth
                summary_writer.add_histogram(tag + '_ground_truth0_distro', ground_truth, global_step=step)

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
