import torch
import net_utils


'''
Encoder architectures
'''
class ResNetEncoder(torch.nn.Module):
    '''
    ResNet encoder with skip connections

    Arg(s):
        n_layer : int
            architecture type based on layers: 18, 34, 50
        input_channels : int
            number of channels in input data
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
    '''

    def __init__(self,
                 n_layer,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(ResNetEncoder, self).__init__()

        use_bottleneck = False
        if n_layer == 18:
            n_blocks = [2, 2, 2, 2]
            resnet_block = net_utils.ResNetBlock
        elif n_layer == 34:
            n_blocks = [3, 4, 6, 3]
            resnet_block = net_utils.ResNetBlock
        elif n_layer == 50:
            n_blocks = [3, 4, 6, 3]
            use_bottleneck = True
            resnet_block = net_utils.ResNetBottleneckBlock
        else:
            raise ValueError('Only supports 18, 34, 50 layer architecture')

        for n in range(len(n_filters) - len(n_blocks) - 1):
            n_blocks = n_blocks + [n_blocks[-1]]

        assert len(n_filters) == len(n_blocks) + 1

        # Keep track on current block
        block_idx = 0
        filter_idx = 0

        activation_func = net_utils.activation_func(activation_func)

        in_channels, out_channels = [input_channels, n_filters[filter_idx]]

        # Resolution 1/1 -> 1/2
        self.conv1 = net_utils.Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        # Resolution 1/2 -> 1/4
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        filter_idx = filter_idx + 1

        blocks2 = []
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):
            if n == 0:
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm)

            blocks2.append(block)

        self.blocks2 = torch.nn.Sequential(*blocks2)

        # Resolution 1/4 -> 1/8
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        blocks3 = []
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):
            if n == 0:
                in_channels = 4 * in_channels if use_bottleneck else in_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=2,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm)

            blocks3.append(block)

        self.blocks3 = torch.nn.Sequential(*blocks3)

        # Resolution 1/8 -> 1/16
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        blocks4 = []
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):
            if n == 0:
                in_channels = 4 * in_channels if use_bottleneck else in_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=2,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm)

            blocks4.append(block)

        self.blocks4 = torch.nn.Sequential(*blocks4)

        # Resolution 1/16 -> 1/32
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        blocks5 = []
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):
            if n == 0:
                in_channels = 4 * in_channels if use_bottleneck else in_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=2,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(
                    in_channels,
                    out_channels,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm)

            blocks5.append(block)

        self.blocks5 = torch.nn.Sequential(*blocks5)

        # Resolution 1/32 -> 1/64
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            blocks6 = []
            in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
            for n in range(n_blocks[block_idx]):
                if n == 0:
                    in_channels = 4 * in_channels if use_bottleneck else in_channels
                    block = resnet_block(
                        in_channels,
                        out_channels,
                        stride=2,
                        weight_initializer=weight_initializer,
                        activation_func=activation_func,
                        use_batch_norm=use_batch_norm,
                        use_instance_norm=use_instance_norm)
                else:
                    in_channels = 4 * out_channels if use_bottleneck else out_channels
                    block = resnet_block(
                        in_channels,
                        out_channels,
                        stride=1,
                        weight_initializer=weight_initializer,
                        activation_func=activation_func,
                        use_batch_norm=use_batch_norm,
                        use_instance_norm=use_instance_norm)

                blocks6.append(block)

            self.blocks6 = torch.nn.Sequential(*blocks6)
        else:
            self.blocks6 = None

        # Resolution 1/64 -> 1/128
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            blocks7 = []
            in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
            for n in range(n_blocks[block_idx]):
                if n == 0:
                    in_channels = 4 * in_channels if use_bottleneck else in_channels
                    block = resnet_block(
                        in_channels,
                        out_channels,
                        stride=2,
                        weight_initializer=weight_initializer,
                        activation_func=activation_func,
                        use_batch_norm=use_batch_norm,
                        use_instance_norm=use_instance_norm)
                else:
                    in_channels = 4 * out_channels if use_bottleneck else out_channels
                    block = resnet_block(
                        in_channels,
                        out_channels,
                        stride=1,
                        weight_initializer=weight_initializer,
                        activation_func=activation_func,
                        use_batch_norm=use_batch_norm,
                        use_instance_norm=use_instance_norm)

                blocks7.append(block)

            self.blocks7 = torch.nn.Sequential(*blocks7)
        else:
            self.blocks7 = None

    def forward(self, x):
        '''
        Forward input x through a ResNet encoder

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        max_pool = self.max_pool(layers[-1])
        layers.append(self.blocks2(max_pool))

        # Resolution 1/4 -> 1/8
        layers.append(self.blocks3(layers[-1]))

        # Resolution 1/8 -> 1/16
        layers.append(self.blocks4(layers[-1]))

        # Resolution 1/16 -> 1/32
        layers.append(self.blocks5(layers[-1]))

        # Resolution 1/32 -> 1/64
        if self.blocks6 is not None:
            layers.append(self.blocks6(layers[-1]))

        # Resolution 1/64 -> 1/128
        if self.blocks7 is not None:
            layers.append(self.blocks7(layers[-1]))

        return layers[-1], layers[1:-1]


class VGGNetEncoder(torch.nn.Module):
    '''
    VGGNet encoder with skip connections

    Arg(s):
        input_channels : int
            number of channels in input data
        n_layer : int
            architecture type based on layers: 8, 11, 13
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
    '''

    def __init__(self,
                 n_layer,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(VGGNetEncoder, self).__init__()

        if n_layer == 8:
            n_convolutions = [1, 1, 1, 1, 1]
        elif n_layer == 11:
            n_convolutions = [1, 1, 2, 2, 2]
        elif n_layer == 13:
            n_convolutions = [2, 2, 2, 2, 2]
        else:
            raise ValueError('Only supports 8, 11, 13 layer architecture')

        for n in range(len(n_filters) - len(n_convolutions) - 1):
            n_convolutions = n_convolutions + [n_convolutions[-1]]

        # Keep track on current block
        block_idx = 0
        filter_idx = 0

        assert len(n_filters) == len(n_convolutions)

        activation_func = net_utils.activation_func(activation_func)

        # Resolution 1/1 -> 1/2
        stride = 1 if n_convolutions[block_idx] - 1 > 0 else 2
        in_channels, out_channels = [input_channels, n_filters[filter_idx]]

        conv1 = net_utils.Conv2d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        if n_convolutions[block_idx] - 1 > 0:
            self.conv1 = torch.nn.Sequential(
                conv1,
                net_utils.VGGNetBlock(
                    out_channels,
                    out_channels,
                    n_convolution=n_convolutions[filter_idx] - 1,
                    stride=2,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm))
        else:
            self.conv1 = conv1

        # Resolution 1/2 -> 1/4
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.conv2 = net_utils.VGGNetBlock(
            in_channels,
            out_channels,
            n_convolution=n_convolutions[block_idx],
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        # Resolution 1/4 -> 1/8
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.conv3 = net_utils.VGGNetBlock(
            in_channels,
            out_channels,
            n_convolution=n_convolutions[block_idx],
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        # Resolution 1/8 -> 1/16
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.conv4 = net_utils.VGGNetBlock(
            in_channels,
            out_channels,
            n_convolution=n_convolutions[block_idx],
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        # Resolution 1/16 -> 1/32
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

        self.conv5 = net_utils.VGGNetBlock(
            in_channels,
            out_channels,
            n_convolution=n_convolutions[block_idx],
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        # Resolution 1/32 -> 1/64
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

            self.conv6 = net_utils.VGGNetBlock(
                in_channels,
                out_channels,
                n_convolution=n_convolutions[block_idx],
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
        else:
            self.conv6 = None

        # Resolution 1/64 -> 1/128
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]

            self.conv7 = net_utils.VGGNetBlock(
                in_channels,
                out_channels,
                n_convolution=n_convolutions[block_idx],
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
        else:
            self.conv7 = None

    def forward(self, x):
        '''
        Forward input x through a VGGNet encoder

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        layers.append(self.conv2(layers[-1]))

        # Resolution 1/4 -> 1/8
        layers.append(self.conv3(layers[-1]))

        # Resolution 1/8 -> 1/32
        layers.append(self.conv4(layers[-1]))

        # Resolution 1/16 -> 1/32
        layers.append(self.conv5(layers[-1]))

        # Resolution 1/32 -> 1/64
        if self.conv6 is not None:
            layers.append(self.conv6(layers[-1]))

        # Resolution 1/64 -> 1/128
        if self.conv7 is not None:
            layers.append(self.conv7(layers[-1]))

        return layers[-1], layers[1:-1]


class PoseEncoder(torch.nn.Module):
    '''
    Pose network encoder

    Arg(s):
        input_channels : int
            number of channels in input data
        n_filters : list[int]
            number of filters to use for each convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
    '''

    def __init__(self,
                 input_channels=6,
                 n_filters=[16, 32, 64, 128, 256, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(PoseEncoder, self).__init__()

        activation_func = net_utils.activation_func(activation_func)

        self.conv1 = net_utils.Conv2d(
            input_channels,
            n_filters[0],
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv2 = net_utils.Conv2d(
            n_filters[0],
            n_filters[1],
            kernel_size=5,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv3 = net_utils.Conv2d(
            n_filters[1],
            n_filters[2],
            kernel_size=3,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv4 = net_utils.Conv2d(
            n_filters[2],
            n_filters[3],
            kernel_size=3,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv5 = net_utils.Conv2d(
            n_filters[3],
            n_filters[4],
            kernel_size=3,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv6 = net_utils.Conv2d(
            n_filters[4],
            n_filters[5],
            kernel_size=3,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv7 = net_utils.Conv2d(
            n_filters[5],
            n_filters[6],
            kernel_size=3,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

    def forward(self, x):
        '''
        Forward input x through a VGGNet encoder

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        layers.append(self.conv2(layers[-1]))

        # Resolution 1/4 -> 1/8
        layers.append(self.conv3(layers[-1]))

        # Resolution 1/8 -> 1/16
        layers.append(self.conv4(layers[-1]))

        # Resolution 1/16 -> 1/32
        layers.append(self.conv5(layers[-1]))

        # Resolution 1/32 -> 1/64
        layers.append(self.conv6(layers[-1]))

        # Resolution 1/64 -> 1/128
        layers.append(self.conv7(layers[-1]))

        return layers[-1], None


'''
Decoder architectures
'''
class MultiScaleDecoder(torch.nn.Module):
    '''
    Multi-scale decoder with skip connections

    Arg(s):
        input_channels : int
            number of channels in input latent vector
        output_channels : int
            number of channels or classes in output
        n_resolution : int
            number of output resolutions (scales) for multi-scale prediction
        n_filters : list[int]
            number of filters to use at each decoder block
        n_skips : list[int]
            number of filters from skip connections
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        output_func : func
            activation function for output
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
        deconv_type : str
            deconvolution types available: transpose, up
    '''

    def __init__(self,
                 input_channels=256,
                 output_channels=1,
                 n_resolution=4,
                 n_filters=[256, 128, 64, 32, 16],
                 n_skips=[256, 128, 64, 32, 0],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 output_func='linear',
                 use_batch_norm=False,
                 use_instance_norm=False,
                 deconv_type='transpose'):
        super(MultiScaleDecoder, self).__init__()

        network_depth = len(n_filters)

        assert network_depth < 8, 'Does not support network depth of 8 or more'
        assert n_resolution > 0 and n_resolution < network_depth

        self.n_resolution = n_resolution
        self.output_func = output_func

        activation_func = net_utils.activation_func(activation_func)
        output_func = net_utils.activation_func(output_func)

        # Upsampling from lower to full resolution requires multi-scale
        if 'upsample' in self.output_func and self.n_resolution < 2:
            self.n_resolution = 2

        filter_idx = 0

        in_channels, skip_channels, out_channels = [
            input_channels, n_skips[filter_idx], n_filters[filter_idx]
        ]

        # Resolution 1/128 -> 1/64
        if network_depth > 6:
            self.deconv6 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm,
                deconv_type=deconv_type)

            filter_idx = filter_idx + 1

            in_channels, skip_channels, out_channels = [
                n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
            ]
        else:
            self.deconv6 = None

        # Resolution 1/64 -> 1/32
        if network_depth > 5:
            self.deconv5 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm,
                deconv_type=deconv_type)

            filter_idx = filter_idx + 1

            in_channels, skip_channels, out_channels = [
                n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
            ]
        else:
            self.deconv5 = None

        # Resolution 1/32 -> 1/16
        if network_depth > 4:
            self.deconv4 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm,
                deconv_type=deconv_type)

            filter_idx = filter_idx + 1

            in_channels, skip_channels, out_channels = [
                n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
            ]
        else:
            self.deconv4 = None

        # Resolution 1/16 -> 1/8
        if network_depth > 3:
            self.deconv3 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm,
                deconv_type=deconv_type)

            if self.n_resolution > 3:
                self.output3 = net_utils.Conv2d(
                    out_channels,
                    output_channels,
                    kernel_size=3,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=None,
                    use_batch_norm=False,
                    use_instance_norm=False)
            else:
                self.output3 = None

            # Resolution 1/8 -> 1/4
            filter_idx = filter_idx + 1

            in_channels, skip_channels, out_channels = [
                n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
            ]

            if self.n_resolution > 3:
                skip_channels = skip_channels + output_channels
        else:
            self.deconv3 = None

        if network_depth > 2:
            self.deconv2 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm,
                deconv_type=deconv_type)

            if self.n_resolution > 2:
                self.output2 = net_utils.Conv2d(
                    out_channels,
                    output_channels,
                    kernel_size=3,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=output_func,
                    use_batch_norm=False,
                    use_instance_norm=False)
            else:
                self.output2 = None

            # Resolution 1/4 -> 1/2
            filter_idx = filter_idx + 1

            in_channels, skip_channels, out_channels = [
                n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
            ]

            if self.n_resolution > 2:
                skip_channels = skip_channels + output_channels
        else:
            self.deconv2 = None

        self.deconv1 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,
            deconv_type=deconv_type)

        if self.n_resolution > 1:
            self.output1 = net_utils.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=output_func,
                use_batch_norm=False,
                use_instance_norm=False)
        else:
            self.output1 = None

        # Resolution 1/2 -> 1/1
        filter_idx = filter_idx + 1

        in_channels, skip_channels, out_channels = [
            n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
        ]

        if self.n_resolution > 1:
            skip_channels = skip_channels + output_channels

        self.deconv0 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm,
            deconv_type=deconv_type)

        self.output0 = net_utils.Conv2d(
            out_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=output_func,
            use_batch_norm=False,
            use_instance_norm=False)

    def forward(self, x, skips, shape=None):
        '''
        Forward latent vector x through decoder network

        Arg(s):
            x : torch.Tensor[float32]
                latent vector
            skips : list[torch.Tensor[float32]]
                list of skip connection tensors (earlier are larger resolution)
            shape : tuple[int]
                (height, width) tuple denoting output size
        Returns:
            list[torch.Tensor[float32]] : list of outputs at multiple scales
        '''

        layers = [x]
        outputs = []

        # Start at the end and walk backwards through skip connections
        n = len(skips) - 1

        # Resolution 1/128 -> 1/64
        if self.deconv6 is not None:
            layers.append(self.deconv6(layers[-1], skips[n]))
            n = n - 1

        # Resolution 1/64 -> 1/32
        if self.deconv5 is not None:
            layers.append(self.deconv5(layers[-1], skips[n]))
            n = n - 1

        # Resolution 1/32 -> 1/16
        if self.deconv4 is not None:
            layers.append(self.deconv4(layers[-1], skips[n]))
            n = n - 1

        # Resolution 1/16 -> 1/8
        if self.deconv3 is not None:
            layers.append(self.deconv3(layers[-1], skips[n]))

            if self.n_resolution > 3:
                output3 = self.output3(layers[-1])
                outputs.append(output3)

                if n > 0:
                    upsample_output3 = torch.nn.functional.interpolate(
                        input=outputs[-1],
                        size=skips[n-1].shape[-2:],
                        mode='bilinear',
                        align_corners=True)
                else:
                    upsample_output3 = torch.nn.functional.interpolate(
                        input=outputs[-1],
                        scale_factor=2,
                        mode='bilinear',
                        align_corners=True)

            n = n - 1

        # Resolution 1/8 -> 1/4
        if self.deconv2 is not None:
            if skips[n] is not None:
                skip = torch.cat([skips[n], upsample_output3], dim=1) if self.n_resolution > 3 else skips[n]
            else:
                skip = skips[n]
            layers.append(self.deconv2(layers[-1], skip))

            if self.n_resolution > 2:
                output2 = self.output2(layers[-1])
                outputs.append(output2)

                if n > 0:
                    upsample_output2 = torch.nn.functional.interpolate(
                        input=outputs[-1],
                        size=skips[n-1].shape[-2:],
                        mode='bilinear',
                        align_corners=True)
                else:
                    upsample_output2 = torch.nn.functional.interpolate(
                        input=outputs[-1],
                        scale_factor=2,
                        mode='bilinear',
                        align_corners=True)

            n = n - 1

        # Resolution 1/4 -> 1/2
        if skips[n] is not None:
            skip = torch.cat([skips[n], upsample_output2], dim=1) if self.n_resolution > 2 else skips[n]
        else:
            skip = skips[n]
        layers.append(self.deconv1(layers[-1], skip))

        if self.n_resolution > 1:
            output1 = self.output1(layers[-1])
            outputs.append(output1)

            if n > 0:
                upsample_output1 = torch.nn.functional.interpolate(
                    input=outputs[-1],
                    size=skips[n-1].shape[-2:],
                    mode='bilinear',
                    align_corners=True)
            else:
                upsample_output1 = torch.nn.functional.interpolate(
                    input=outputs[-1],
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True)

        # Resolution 1/2 -> 1/1
        n = n - 1

        if 'upsample' in self.output_func:
            output0 = upsample_output1
        else:
            if self.n_resolution > 1:
                # If there is skip connection at layer 0
                if skips[n] is not None and n == 0:
                    skip = torch.cat([skips[n], upsample_output1], dim=1) if n == 0 else upsample_output1
                else:
                    skip = upsample_output1
                layers.append(self.deconv0(layers[-1], skip))
            else:

                if skips[n] is not None and n == 0:
                    layers.append(self.deconv0(layers[-1], skips[n]))
                else:
                    layers.append(self.deconv0(layers[-1], shape=shape[-2:]))

            output0 = self.output0(layers[-1])

        outputs.append(output0)

        return outputs


class PoseDecoder(torch.nn.Module):
    '''
    Pose Decoder 6 DOF

    Arg(s):
        rotation_parameterization : str
            axis
        input_channels : int
            number of channels in input latent vector
        n_filters : int list
            number of filters to use at each decoder block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
    '''

    def __init__(self,
                 rotation_parameterization,
                 input_channels=256,
                 n_filters=[],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(PoseDecoder, self).__init__()

        self.rotation_parameterization = rotation_parameterization

        activation_func = net_utils.activation_func(activation_func)

        if len(n_filters) > 0:
            layers = []
            in_channels = input_channels

            for out_channels in n_filters:
                conv = net_utils.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm,
                    use_instance_norm=use_instance_norm)
                layers.append(conv)
                in_channels = out_channels

            conv = net_utils.Conv2d(
                in_channels=in_channels,
                out_channels=6,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=None,
                use_batch_norm=False,
                use_instance_norm=False)
            layers.append(conv)

            self.conv = torch.nn.Sequential(*layers)
        else:
            self.conv = net_utils.Conv2d(
                in_channels=input_channels,
                out_channels=6,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=None,
                use_batch_norm=False,
                use_instance_norm=False)

    def forward(self, x):
        conv_output = self.conv(x)
        pose_mean = torch.mean(conv_output, [2, 3])
        dof = 0.01 * pose_mean
        posemat = net_utils.pose_matrix(
            dof,
            rotation_parameterization=self.rotation_parameterization)

        return posemat


class SpatialPyramidPool(torch.nn.Module):
    '''
    Max pool with different kernel sizes (small gives detail, large gives density)
        and combines them with 1 x 1 convolutions

    Arg(s):
        input_channels : int
            number of channels to be fed to max pool(s)
        pool_sizes : list[int]
            list of max pool sizes s (kernel size is s x s)
        n_convolution : int
            number of 1 x 1 convolutions to use for balancing detail and density
        n_filter : int
            number of filters for 1 x 1 convolutions
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
    '''

    def __init__(self,
                 input_channels,
                 pool_sizes=[3, 5, 7, 9],
                 n_convolution=3,
                 n_filter=8,
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu'):
        super(SpatialPyramidPool, self).__init__()

        activation_func = net_utils.activation_func(activation_func)

        self.pools = []
        for s in pool_sizes:
            padding = s // 2
            pool = torch.nn.MaxPool2d(kernel_size=s, stride=1, padding=padding)
            self.pools.append(pool)

        convs = []
        for n in range(n_convolution):
            conv = net_utils.Conv2d(
                input_channels,
                n_filter,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=False)
            convs.append(conv)

            # Set new input channels as output channels
            input_channels = n_filter

        self.convs = torch.nn.Sequential(*convs)

    def forward(self, x):
        layers = [x]

        for pool in self.pools:
            layers.append(pool(x))

        # Stack pools into pyramid
        pyramid = torch.cat(layers, dim=1)

        # Convolutions to balance trade-off between detail and density
        return self.convs(pyramid)
