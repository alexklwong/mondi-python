import torch
import numpy as np


EPSILON = 1e-10


def activation_func(activation_fn):
    '''
    Select activation function

    Arg(s):
        activation_fn : str
            name of activation function
    Returns:
        torch.nn.Module : activation function
    '''

    if 'linear' in activation_fn:
        return None
    elif 'leaky_relu' in activation_fn:
        return torch.nn.LeakyReLU(negative_slope=0.20, inplace=True)
    elif 'relu' in activation_fn:
        return torch.nn.ReLU()
    elif 'elu' in activation_fn:
        return torch.nn.ELU()
    elif 'sigmoid' in activation_fn:
        return torch.nn.Sigmoid()
    else:
        raise ValueError('Unsupported activation function: {}'.format(activation_fn))


'''
Network layers
'''
class Conv2d(torch.nn.Module):
    '''
    2D convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel
        stride : int
            stride of convolution
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
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(Conv2d, self).__init__()

        padding = kernel_size // 2

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.activation_func = activation_func

        assert not (use_batch_norm and use_instance_norm), \
            'Unable to apply both batch and instance normalization'

        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        '''
        Forward input x through a convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        conv = self.conv(x)

        if self.use_batch_norm:
            conv = self.batch_norm(conv)
        elif self.use_instance_norm:
            conv = self.instance_norm(conv)

        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv


class TransposeConv2d(torch.nn.Module):
    '''
    Transpose convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel (k x k)
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
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(TransposeConv2d, self).__init__()

        padding = kernel_size // 2

        self.deconv = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.deconv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.deconv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.deconv.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.activation_func = activation_func

        assert not (use_batch_norm and use_instance_norm), \
            'Unable to apply both batch and instance normalization'

        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        '''
        Forward input x through a transposed convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        '''

        deconv = self.deconv(x)

        if self.use_batch_norm:
            deconv = self.batch_norm(deconv)
        elif self.use_instance_norm:
            deconv = self.instance_norm(deconv)

        if self.activation_func is not None:
            return self.activation_func(deconv)
        else:
            return deconv


class UpConv2d(torch.nn.Module):
    '''
    Up-convolution (upsample + convolution) block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        shape : list[int]
            two element tuple of ints (height, width)
        kernel_size : int
            size of kernel (k x k)
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
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(UpConv2d, self).__init__()

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

    def forward(self, x, shape):
        '''
        Forward input x through an up convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
            shape : tuple[int]
                height, width (H, W) tuple denoting output shape
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        '''

        upsample = torch.nn.functional.interpolate(x, size=shape, mode='nearest')
        conv = self.conv(upsample)
        return conv


'''
Network encoder blocks
'''
class ResNetBlock(torch.nn.Module):
    '''
    Basic ResNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        stride : int
            stride of convolution
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
                 in_channels,
                 out_channels,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(ResNetBlock, self).__init__()

        self.activation_func = activation_func

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.projection = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=False,
            use_instance_norm=False)

    def forward(self, x):
        '''
        Forward input x through a basic ResNet block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        # Perform 2 convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        # Perform projection if (1) shape does not match (2) channels do not match
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)
        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x

        # f(x) + x
        return self.activation_func(conv2 + X)


class ResNetBottleneckBlock(torch.nn.Module):
    '''
    ResNet bottleneck block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        stride : int
            stride of convolution
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
                 in_channels,
                 out_channels,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(ResNetBottleneckBlock, self).__init__()

        self.activation_func = activation_func

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv3 = Conv2d(
            out_channels,
            4 * out_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.projection = Conv2d(
            in_channels,
            4 * out_channels,
            kernel_size=1,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=False,
            use_instance_norm=False)

    def forward(self, x):
        '''
        Forward input x through a ResNet bottleneck block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        # Perform 2 convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # Perform projection if (1) shape does not match (2) channels do not match
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)
        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x

        # f(x) + x
        return self.activation_func(conv3 + X)


class VGGNetBlock(torch.nn.Module):
    '''
    VGGNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        n_convolution : int
            number of convolution layers
        stride : int
            stride of convolution
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
                 in_channels,
                 out_channels,
                 n_convolution=1,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(VGGNetBlock, self).__init__()

        layers = []
        for n in range(n_convolution - 1):
            conv = Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
            layers.append(conv)
            in_channels = out_channels

        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)
        layers.append(conv)

        self.conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        '''
        Forward input x through a VGG block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        return self.conv_block(x)


'''
Network decoder blocks
'''
class DecoderBlock(torch.nn.Module):
    '''
    Decoder block with skip connection

    Arg(s):
        in_channels : int
            number of input channels
        skip_channels : int
            number of skip connection channels
        out_channels : int
            number of output channels
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        use_instance_norm : bool
            if set, then applied instance normalization
        deconv_type : str
            deconvolution types: transpose, up
    '''

    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 deconv_type='up'):
        super(DecoderBlock, self).__init__()

        self.skip_channels = skip_channels
        self.deconv_type = deconv_type

        if deconv_type == 'transpose':
            self.deconv = TransposeConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
        elif deconv_type == 'up':
            self.deconv = UpConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)

        concat_channels = skip_channels + out_channels

        self.conv = Conv2d(
            concat_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

    def forward(self, x, skip=None, shape=None):
        '''
        Forward input x through a decoder block and fuse with skip connection

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
            skip : torch.Tensor[float32]
                N x F x H x W skip connection
            shape : tuple[int]
                height, width (H, W) tuple denoting output shape
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        '''

        if self.deconv_type == 'transpose':
            deconv = self.deconv(x)
        elif self.deconv_type == 'up':

            if skip is not None:
                shape = skip.shape[2:4]
            elif shape is not None:
                pass
            else:
                n_height, n_width = x.shape[2:4]
                shape = (int(2 * n_height), int(2 * n_width))

            deconv = self.deconv(x, shape=shape)
            # print(shape)
        if self.skip_channels > 0:
            concat = torch.cat([deconv, skip], dim=1)
        else:
            concat = deconv

        return self.conv(concat)


'''
Pose regression layer
'''
def pose_matrix(v, rotation_parameterization='axis'):
    '''
    Convert 6 DoF parameters to transformation matrix

    Arg(s):
        v : torch.Tensor[float32]
            N x 6 vector in the order of tx, ty, tz, rx, ry, rz
        rotation_parameterization : str
            axis
    Returns:
        torch.Tensor[float32] : N x 4 x 4 homogeneous transformation matrix
    '''

    # Select N x 3 element rotation vector
    r = v[..., :3]
    # Select N x 3 element translation vector
    t = v[..., 3:]

    if rotation_parameterization == 'axis':
        Rt = transformation_from_parameters(torch.unsqueeze(r, dim=1), t)
    else:
        raise ValueError('Unsupported rotation parameterization: {}'.format(rotation_parameterization))

    return Rt


'''
Utility functions for rotation
'''
def tilde(v):
    '''
    Tilde (hat) operation

    Arg(s):
        v : torch.Tensor[float32]
            3 element rotation vector
    Returns:
        torch.Tensor[float32] : 3 x 3 skew matrix
    '''

    v1, v2, v3 = v[0], v[1], v[2]
    zero = torch.tensor(0.0, device=v.device)
    r1 = torch.stack([zero,  -v3,   v2], dim=0)
    r2 = torch.stack([  v3, zero,  -v1], dim=0)
    r3 = torch.stack([ -v2,   v1, zero], dim=0)

    return torch.stack([r1, r2, r3], dim=0)

def tilde_inv(R):
    '''
    Inverse of the tilde operation

    Arg(s):
        R : torch.Tensor[float32]
            3 x 3 inverse skew matrix
    Returns:
        torch.Tensor[float32] : 3 element rotation vector
    '''

    return 0.5 * torch.stack([R[2, 1] - R[1, 2],
                              R[0, 2] - R[2, 0],
                              R[1, 0] - R[0, 1]], dim=0)

def log_map(R):
    '''
    Logarithm map of rotation matrix element

    Arg(s):
        R : torch.Tensor[float32]
            3 x 3 rotation matrix
    Returns:
        torch.Tensor[float32] : 3 element rotation vector
    '''

    trR = 0.5 * (torch.trace(R) - 1.0)

    if trR >= 1.0:
        return tilde_inv(R)
    else:
        th = torch.acos(trR)
        return tilde_inv(R) * (th / torch.sin(th))

def exp_map(v):
    '''
    Exponential map of rotation vector using Rodrigues

    Arg(s):
        v : torch.Tensor[float32]
            3 element rotation vector
    Returns:
        torch.Tensor[float32] : 3 x 3 rotation matrix
    '''

    device = v.device
    th = torch.norm(v, p='fro')

    if th < EPSILON:
        return tilde(v) + torch.eye(3, device=device)
    else:
        sin_th = torch.sin(th)
        cos_th = torch.cos(th)
        W = tilde(v / th)
        WW = torch.matmul(W, W)
        R = sin_th * W + (1.0 - cos_th) * WW
        return R + torch.eye(3, device=device)

def batch_log_map(R):
    '''
    Applies logarithmic map for entire batch

    Arg(s):
        R : torch.Tensor[float32]
            N x 3 x 3 rotation matrices
    Returns:
        torch.Tensor[float32] : N x 3 vectors
    '''

    outputs = []
    n_batch = R.shape[0]
    for n in range(n_batch):
        outputs.append(log_map(R[n, :, :]))

    return torch.stack(outputs, dim=0)

def batch_exp_map(v):
    '''
    Applies exponential map for entire batch

    Arg(s):
        v : torch.Tensor[float32]
            N x 3 vectors
    Returns:
        torch.Tensor[float32] : N x 3 x 3 rotation matrices
    '''

    outputs = []
    n_batch = v.shape[0]
    for n in range(n_batch):
        outputs.append(exp_map(v[n, :]))

    return torch.stack(outputs, dim=0)

def euler_angles(v):
    '''
    Converts euler angles to rotation matrix

    Arg(s):
        v : torch.Tensor[float32]
            N x 3 vector of rotation angles along x, y, z axis
    Returns:
        torch.Tensor[float32] : N x 3 x 3 rotation matrix corresponding to the euler angles
    '''

    n_batch = v.shape[0]
    # Expand to B x 1 x 1
    x = torch.clamp(v[:, 0], min=-np.pi, max=np.pi).view(n_batch, 1, 1)
    y = torch.clamp(v[:, 1], min=-np.pi, max=np.pi).view(n_batch, 1, 1)
    z = torch.clamp(v[:, 2], min=-np.pi, max=np.pi).view(n_batch, 1, 1)

    zeros = torch.zeros_like(z)
    ones = torch.ones_like(z)

    # Rotation about x-axis
    cos_x = torch.cos(x)
    sin_x = torch.sin(x)
    rot_mat_x_row1 = torch.cat([ ones, zeros,  zeros], dim=2)
    rot_mat_x_row2 = torch.cat([zeros, cos_x, -sin_x], dim=2)
    rot_mat_x_row3 = torch.cat([zeros, sin_x,  cos_x], dim=2)
    rot_mat_x = torch.cat([rot_mat_x_row1, rot_mat_x_row2, rot_mat_x_row3], dim=1)

    # Rotation about y-axis
    cos_y = torch.cos(y)
    sin_y = torch.sin(y)
    rot_mat_y_row1 = torch.cat([ cos_y, zeros, sin_y], dim=2)
    rot_mat_y_row2 = torch.cat([ zeros,  ones, zeros], dim=2)
    rot_mat_y_row3 = torch.cat([-sin_y, zeros, cos_y], dim=2)
    rot_mat_y = torch.cat([rot_mat_y_row1, rot_mat_y_row2, rot_mat_y_row3], dim=1)

    # Rotation about z-axis
    cos_z = torch.cos(z)
    sin_z = torch.sin(z)
    rot_mat_z_row1 = torch.cat([cos_z, -sin_z, zeros], dim=2)
    rot_mat_z_row2 = torch.cat([sin_z,  cos_z, zeros], dim=2)
    rot_mat_z_row3 = torch.cat([zeros,  zeros,  ones], dim=2)
    rot_mat_z = torch.cat([rot_mat_z_row1, rot_mat_z_row2, rot_mat_z_row3], dim=1)

    return torch.matmul(torch.matmul(rot_mat_x, rot_mat_y), rot_mat_z)

def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M

def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


'''
Utility functions for rigid warping
'''
def meshgrid(n_batch, n_height, n_width, device, homogeneous=True):
    '''
    Creates N x 2 x H x W meshgrid from -1 to 1 in x, y directions

    Arg(s):
        n_batch : int
            batch size
        n_height : int
            height of tensor
        n_width : int
            width of tensor
        device : torch.device
            device on which to create meshgrid
        homoegenous : bool
            if set, then add homogeneous coordinates (N x H x W x 3)
    Return:
        torch.Tensor[float32] : N x 2 x H x W meshgrid of x (dim 0), y (dim 1) and 1 (dim 2, if homogeneous)
    '''

    x = torch.linspace(start=0.0, end=n_width-1, steps=n_width, device=device)
    y = torch.linspace(start=0.0, end=n_height-1, steps=n_height, device=device)
    # Create H x W grids
    grid_y, grid_x = torch.meshgrid(y, x)

    if homogeneous:
        # Create 3 x H x W grid (x, y, 1)
        grid_xy = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0)
    else:
        # Create 2 x H x W grid (x, y)
        grid_xy = torch.stack([grid_x, grid_y], dim=0)

    grid_xy = torch.unsqueeze(grid_xy, dim=0) \
        .repeat(n_batch, 1, 1, 1)
    return grid_xy

def backproject_to_camera(depth, camera, shape):
    '''
    Backprojects pixel coordinates to 3D camera coordinates

    Arg(s):
        depth : torch.Tensor[float32]
            N x 1 x H x W depth map
        camera : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics
        shape : list
            shape of tensor in (N, C, H, W)
    Return:
        torch.Tensor[float32] : N x 4 x (H x W)
    '''

    n_batch, _, n_height, n_width = shape

    # Create homogeneous coordinates [x, y, 1]
    xy_h = meshgrid(n_batch, n_height, n_width, device=depth.device, homogeneous=True)

    # Reshape pixel coordinates to N x 3 x (H x W)
    xy_h = xy_h.view(n_batch, 3, -1)

    # Reshape depth as N x 1 x (H x W)
    depth = depth.view(n_batch, 1, -1)

    # K^-1 [x, y, 1] z
    points = torch.matmul(torch.inverse(camera), xy_h * depth)

    # Make homogeneous
    return torch.cat([points, torch.ones_like(depth)], dim=1)

def project_to_pixel(points, pose, camera, shape):
    '''
    Projects points in camera coordinates to 2D pixel coordinates

    Arg(s):
        points : torch.Tensor[float32]
            N x 4 x (H x W) depth map
        pose : torch.Tensor[float32]
            N x 4 x 4 transformation matrix
        camera : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics
        shape : list
            shape of tensor in (N, C, H, W)
    Return:
        torch.Tensor[float32] : N x 2 x H x W
    '''

    n_batch, _, n_height, n_width = shape

    # Convert camera intrinsics to homogeneous coordinates
    column = torch.zeros([n_batch, 3, 1], device=points.device)
    row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=points.device) \
        .view(1, 1, 4) \
        .repeat(n_batch, 1, 1)
    camera = torch.cat([camera, column], dim=2)
    camera = torch.cat([camera, row], dim=1)

    # Apply the transformation and project: \pi K g p
    T = torch.matmul(camera, pose)
    points = torch.matmul(T, points)
    points = points / (torch.unsqueeze(points[:, 2, :], dim=1) + 1e-10)
    points = points[:, 0:2, :]

    # Reshape to N x 2 x H x W
    return points.view(n_batch, 2, n_height, n_width)

def grid_sample(image, target_xy, shape):
    '''
    Samples the image at x, y locations to target x, y locations

    Arg(s):
        image : torch.Tensor[float32]
            N x 3 x H x W RGB image
        target_xy : torch.Tensor[float32]
            N x 2 x H x W target x, y locations in image space
        shape : list[int]
            shape of tensor in (N, C, H, W)
    Return:
        torch.Tensor[float32] : N x 3 x H x W RGB image
    '''

    n_batch, _, n_height, n_width = shape

    # Swap dimensions to N x H x W x 2 for grid sample
    target_xy = target_xy.permute(0, 2, 3, 1)
    # Normalize coordinates between -1 and 1
    target_xy[..., 0] /= (n_width - 1.0)
    target_xy[..., 1] /= (n_height - 1.0)
    target_xy = 2.0 * (target_xy - 0.5)

    # Sample the image at normalized target x, y locations
    return torch.nn.functional.grid_sample(image, target_xy, padding_mode='zeros')


class BackprojectDepth(torch.nn.Module):

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = torch.nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = torch.nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = torch.nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(torch.nn.Module):

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)
        P = P[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


'''
Utility function to pre-process sparse depth and input depth
'''
class OutlierRemoval(object):
    '''
    Class to perform outlier removal based on depth difference in local neighborhood

    Arg(s):
        kernel_size : int
            local neighborhood to consider
        threshold : float
            depth difference threshold
    '''

    def __init__(self, kernel_size=7, threshold=1.5):

        self.kernel_size = kernel_size
        self.threshold = threshold

    def remove_outliers(self, sparse_depth, validity_map):
        '''
        Removes erroneous measurements from sparse depth and validity map

        Arg(s):
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W tensor sparse depth
            validity_map : torch.Tensor[float32]
                N x 1 x H x W tensor validity map
        Returns:
            torch.Tensor[float32] : N x 1 x H x W sparse depth
            torch.Tensor[float32] : N x 1 x H x W validity map
        '''

        # Replace all zeros with large values
        max_value = 10 * torch.max(sparse_depth)
        sparse_depth_max_filled = torch.where(
            validity_map <= 0,
            torch.full_like(sparse_depth, fill_value=max_value),
            sparse_depth)

        # For each neighborhood find the smallest value
        padding = self.kernel_size // 2
        sparse_depth_max_filled = torch.nn.functional.pad(
            input=sparse_depth_max_filled,
            pad=(padding, padding, padding, padding),
            mode='constant',
            value=max_value)

        min_values = -torch.nn.functional.max_pool2d(
            input=-sparse_depth_max_filled,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0)

        # If measurement differs a lot from minimum value then remove
        validity_map_clean = torch.where(
            min_values < sparse_depth - self.threshold,
            torch.zeros_like(validity_map),
            torch.ones_like(validity_map))

        # Update sparse depth and validity map
        validity_map_clean = validity_map * validity_map_clean
        sparse_depth_clean = sparse_depth * validity_map_clean

        return sparse_depth_clean, validity_map_clean

def scale_match(input_depth, sparse_depth, method='none', kernel_size=1):
    '''
    Matches scale of input depth to sparse depth

    Arg(s):
        input_depth : torch.Tensor[float32]
            N x 1 x H x W input dense depth
        sparse_depth : torch.Tensor[float32]
            N x 1 x H x W sparse depth
        method : str
            scale matching method: replace, local_scale, none
        kernel_size : int
            kernel size for scaling or replacing
    Returns:
        torch.Tensor[float32] : dense depth
    '''

    output_depth = input_depth

    if method == 'replace':
        # Replace input depth values with sparse depth values
        output_depth = torch.where(
            sparse_depth > 0,
            sparse_depth,
            input_depth)
    elif method == 'local_scale':
        # Find scaling factor for each sparse point
        scale = torch.where(
            sparse_depth > 0,
            sparse_depth / input_depth,
            sparse_depth)

        # In case sparse and input are very different
        scale = torch.where(
            scale > 2,
            torch.zeros_like(scale),
            scale)

        # Dilate neighborhood based on kernel size
        padding = kernel_size // 2
        max_pool = torch.nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=padding)
        scale = max_pool(scale)

        # Set all zeros to ones for scaling factors
        scale = torch.where(
            scale > 0,
            scale,
            torch.ones_like(scale))

        # Multiply input depth by scale
        output_depth = scale * input_depth

    return output_depth
