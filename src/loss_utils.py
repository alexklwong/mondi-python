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
import random


'''
Loss functions
'''
def color_consistency_loss_func(src, tgt, w=None, reduce_loss=True):
    '''
    Computes the color consistency loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
        w : torch.Tensor[float32]
            N x 1 x H x W weights
        reduce_loss : bool
            Whether to reduce loss over height and weight dimensions
    Returns:
        Either
        (1) [reduce_loss=True] torch.Tensor[float32] : mean absolute difference between source and target images
        (2) [reduce_loss=False] torch.Tensor[float32] : absolute difference between source and target images, N x 1 x H x W
    '''

    if w is None:
        w = torch.ones_like(src)

    loss = torch.sum(w * torch.abs(tgt - src), dim=1)

    if reduce_loss:
        return torch.mean(loss)
    else:
        return loss.unsqueeze(1)

def structural_consistency_loss_func(src, tgt, w=None, reduce_loss=True):
    '''
    Computes the structural consistency loss using SSIM

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
        w : torch.Tensor[float32]
            N x 3 x H x W weights
        reduce_loss : bool
            if set then return mean over loss
    Returns:
        Either
        (1) [reduce_loss=True] torch.Tensor[float32] : mean absolute difference between source and target images
        (2) [reduce_loss=False] torch.Tensor[float32] : absolute difference between source and target images, N x 1 x H x W
    '''

    if w is None:
        w = torch.ones_like(src)

    refl = torch.nn.ReflectionPad2d(1)

    src = refl(src)
    tgt = refl(tgt)
    scores = ssim(src, tgt)

    loss = torch.sum(w * scores, dim=1)

    if reduce_loss:
        return torch.mean(loss)
    else:
        return loss.unsqueeze(1)

def sparse_depth_consistency_loss_func(src, tgt, w=None):
    '''
    Computes the sparse depth consistency loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 1 x H x W source depth
        tgt : torch.Tensor[float32]
            N x 1 x H x W target depth
        w : torch.Tensor[float32]
            N x 1 x H x W weights
    Returns:
        torch.Tensor[float32] : mean absolute difference between source and target depth
    '''

    if w is None:
        w = torch.ones_like(src)

    delta = torch.abs(tgt - src)
    loss = torch.sum(w * delta, dim=[1, 2, 3])

    return torch.mean(loss / torch.sum(w, dim=[1, 2, 3]))

def smoothness_loss_func(predict, image, weights):
    '''
    Computes the local smoothness loss

    Arg(s):
        predict : torch.Tensor[float32]
            N x 1 x H x W predictions
        image : torch.Tensor[float32]
            N x 3 x H x W RGB image
        weights : torch.Tensor[float32]
            N x 1 x H x W weights of each pixel
    Returns:
        torch.Tensor[float32] : mean SSIM distance between source and target images
    '''

    # Add weights parameter (scalar)
    predict_dy, predict_dx = gradient_yx(predict)
    image_dy, image_dx = gradient_yx(image)

    # Create edge awareness weights
    weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    # Match shape since gradient cuts off last row and column
    weights_x = weights_x * weights[..., :, :-1]
    weights_y = weights_y * weights[..., :-1, :]

    smoothness_x = torch.mean(weights_x * torch.abs(predict_dx))
    smoothness_y = torch.mean(weights_y * torch.abs(predict_dy))

    return smoothness_x + smoothness_y

def aggregate_teacher_output(losses0x, teacher_output0):
    '''
    Combines ensemble of teacher output

    Arg(s):
        losses0x : list[torch.Tensor[float32]]
            list[N x 1 x H x W] losses for each teacher
        teacher_output0 : torch.Tensor[float32]
            N x M x H x W teacher output from ensemble of M teachers
    Returns:
        torch.Tensor[float32] : N x 1 x H x W aggregated teacher output from ensemble
        torch.Tensor[float32] : N x 1 x H x W loss for each selected prediction from teacher
        torch.Tensor[float32] : N x 1 x H x W teacher id (index) for each prediction that minimize loss
    '''

    device = teacher_output0.device
    N, M, H, W = teacher_output0.shape

    if len(losses0x) == 0:
        # Use random by default if no losses are available
        r = random.randint(0, M-1)
        agg_teacher_output0 = teacher_output0[:, r:r+1]

        return agg_teacher_output0, torch.zeros_like(agg_teacher_output0) + 0.05, None

    losses0x_selection = losses0x

    agg_teacher_output = torch.zeros((N, 1, H, W)).to(device)
    teacher_output_loss = torch.zeros((N, 1, H, W)).to(device)
    losses0x_selection = torch.stack(losses0x_selection, dim=0)

    # Select teachers based on ones that minimize losses
    _, teacher_idxs = torch.min(losses0x_selection, dim=0)

    for i in range(M):
        teacher_output_i = teacher_output0[:, i:i+1]

        # Take loss from original
        teacher_output_loss_i = losses0x[i]

        # Create a mask to mask out anything that isn't from selected and insert into output
        teacher_mask = teacher_idxs == i
        agg_teacher_output += teacher_output_i * teacher_mask
        teacher_output_loss += teacher_output_loss_i * teacher_mask

    return agg_teacher_output, teacher_output_loss, teacher_idxs

def smooth_l1_loss(src, tgt, w=None, beta=1.0):
    '''
    Computes smooth_l1 loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
        w : torch.Tensor[float32]
            N x 1 x H x W weights
    '''

    if w is None:
        w = torch.ones_like(src)

    loss_func = torch.nn.SmoothL1Loss(reduction='none')

    loss = loss_func(src, tgt)
    loss = torch.sum(w * loss, dim=[1, 2, 3]) / torch.sum(w, dim=[1, 2, 3])
    return torch.mean(loss)

def l1_loss(src, tgt, w=None):
    '''
    Computes l1 loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
        w : torch.Tensor[float32]
            N x 1 x H x W weights
    Returns:
        float : mean l1 loss across batch

    '''

    if w is None:
        w = torch.ones_like(src)

    loss_func = torch.nn.L1Loss(reduction='none')
    loss = loss_func(src, tgt)
    loss = torch.sum(w * loss, dim=[1, 2, 3]) / torch.sum(w, dim=[1, 2, 3])

    return torch.mean(loss)

def l2_loss(src, tgt, w=None):
    '''
    Computes l2 loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image
        w : torch.Tensor[float32]
            N x 1 x H x W weights
    Returns:
        float : mean l2 loss across batch
    '''

    if w is None:
        w = torch.ones_like(src)

    loss_func = torch.nn.MSELoss(reduction='none')
    loss = loss_func(src, tgt)
    loss = torch.sum(w * loss, dim=[1, 2, 3]) / torch.sum(w, dim=[1, 2, 3])
    return torch.mean(loss)


'''
Helper functions for constructing loss functions
'''
def gradient_yx(T):
    '''
    Computes gradients in the y and x directions

    Arg(s):
        T : torch.Tensor[float32]
            N x C x H x W tensor
    Returns:
        torch.Tensor[float32] : gradients in y direction
        torch.Tensor[float32] : gradients in x direction
    '''

    dx = T[:, :, :, :-1] - T[:, :, :, 1:]
    dy = T[:, :, :-1, :] - T[:, :, 1:, :]
    return dy, dx

def ssim(x, y):
    '''
    Computes Structural Similarity Index distance between two images

    Arg(s):
        x : torch.Tensor[float32]
            N x 3 x H x W RGB image
        y : torch.Tensor[float32]
            N x 3 x H x W RGB image
    Returns:
        torch.Tensor[float32] : SSIM distance between two images
    '''

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = torch.nn.AvgPool2d(3, 1)(x)
    mu_y = torch.nn.AvgPool2d(3, 1)(y)
    mu_xy = mu_x * mu_y
    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2

    sigma_x = torch.nn.AvgPool2d(3, 1)(x ** 2) - mu_xx
    sigma_y = torch.nn.AvgPool2d(3, 1)(y ** 2) - mu_yy
    sigma_xy = torch.nn.AvgPool2d(3, 1)(x * y) - mu_xy

    numer = (2 * mu_xy + C1)*(2 * sigma_xy + C2)
    denom = (mu_xx + mu_yy + C1) * (sigma_x + sigma_y + C2)
    score = numer / denom

    return torch.clamp((1.0 - score) / 2.0, 0.0, 1.0)


'''
Rigid warping functions
'''
def warp1d_horizontal(image, disparity, padding_mode='border'):
    '''
    Performs horizontal 1d warping

    Arg(s):
        image : torch.Tensor[float32]
            N x C x H x W image
        disparity : torch.Tensor[float32]
            N x 1 x H x W disparity
    '''

    n_batch, _, n_height, n_width = image.shape

    # Original coordinates of pixels
    x = torch.linspace(0, 1, n_width, dtype=torch.float32, device=image.device) \
        .repeat(n_batch, n_height, 1)
    y = torch.linspace(0, 1, n_height, dtype=torch.float32, device=image.device) \
        .repeat(n_batch, n_width, 1) \
        .transpose(1, 2)

    # Apply shift in X direction
    dx = disparity[:, 0, :, :] / n_width  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x + dx, y), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    return torch.nn.functional.grid_sample(
        image,
        grid=(2 * flow_field - 1),
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True)

def rigid_warp(image1, depth0, pose01, intrinsics, shape):
    '''
    Rigid warping of image1 to image0 using pose01

    Arg(s):
        image1 : torch.Tensor[float32]
            N x C x H x W image0
        depth0 : torch.Tensor[float32]
            N x 1 x H x W depth of image0
        pose01 : torch.Tensor[float32]
            N x 4 x 4 transformation matrix
        intrinsics : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics
        shape : list[int]
            shape of tensor in (N, C, H, W)
    '''

    points = backproject_to_camera(depth0, intrinsics, shape)
    xy = project_to_pixel(points, pose01, intrinsics, shape)
    warped = grid_sample(image1, xy, shape)
    return warped


'''
Utility functions for rigid warping
'''
def meshgrid(n_batch, n_height, n_width, device, homogeneous=True):
    '''
    Creates N x 2 x H x W meshgrid in x, y directions

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
        torch.Tensor[float32]: N x 2 x H x W meshgrid of x, y and 1 (if homogeneous)
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

def backproject_to_camera(depth, intrinsics, shape):
    '''
    Backprojects pixel coordinates to 3D camera coordinates

    Arg(s):
        depth : torch.Tensor[float32]
            N x 1 x H x W depth map
        intrinsics : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics
        shape : list[int]
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
    points = torch.matmul(torch.inverse(intrinsics), xy_h) * depth

    # Make homogeneous
    return torch.cat([points, torch.ones_like(depth)], dim=1)

def project_to_pixel(points, pose, intrinsics, shape):
    '''
    Projects points in camera coordinates to 2D pixel coordinates

    Arg(s):
        points : torch.Tensor[float32]
            N x 4 x (H x W) depth map
        pose : torch.Tensor[float32]
            N x 4 x 4 transformation matrix
        intrinsics : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics
        shape : list[int]
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
    intrinsics = torch.cat([intrinsics, column], dim=2)
    intrinsics = torch.cat([intrinsics, row], dim=1)

    # Apply the transformation and project: \pi K g p
    T = torch.matmul(intrinsics, pose)
    T = T[:, 0:3, :]
    points = torch.matmul(T, points)
    points = points / (torch.unsqueeze(points[:, 2, :], dim=1) + 1e-7)
    points = points[:, 0:2, :]

    # Reshape to N x 2 x H x W
    return points.view(n_batch, 2, n_height, n_width)

def grid_sample(image, target_xy, shape, padding_mode='border'):
    '''
    Samples the image at x, y locations to target x, y locations

    Arg(s):
        image : torch.Tensor[float32]
            N x 3 x H x W RGB image
        target_xy : torch.Tensor[float32]
            N x 2 x H x W target x, y locations in image space
        shape : list[int]
            shape of tensor in (N, C, H, W)
        padding_mode : str
            padding to use when sampled out of bounds
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
    return torch.nn.functional.grid_sample(
        image,
        grid=target_xy,
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True)

def dilate_sparse_depth(sparse_depth, dilation_kernel_size):
    '''
    Dilates non-zero sparse depth values to kernel size specified

    Arg(s):
        sparse_depth : torch.Tensor[float32]
            N x 1 x H x W sparse depth tensor
        dilation_kernel_size : int
            Odd number of how large of a window to repeat sparse depth values
    Returns:
        torch.Tensor[float32] : N x 1 x H x W tensor of dilated sparse depth values
        torch.Tensor[float32] : N x 1 x H x W binary validity map tensor
    '''

    if dilation_kernel_size < 1 or dilation_kernel_size % 2 == 0:
        raise ValueError('Invalid valiue {} for dilation_kernel_size. Must be odd positive integer.'
            .format(dilation_kernel_size))

    # Dilation kernel size = 1 -> return sparse depth
    if dilation_kernel_size == 1:
        dilated_validity_map = torch.where(sparse_depth > 0, 1, 0)
        return sparse_depth, dilated_validity_map

    # dilated_sparse_depth = torch.zeros_like(sparse_depth)

    # Fill in dilated_sparse_depth
    padding = dilation_kernel_size // 2

    max_pool = torch.nn.MaxPool2d(
        kernel_size=dilation_kernel_size,
        stride=1,
        padding=padding)

    # Perform non-zero min pooling with MaxPool2d function by negating
    dilated_sparse_depth = -max_pool(torch.where(
        sparse_depth == 0,
        -999 * torch.ones_like(sparse_depth),
        -sparse_depth))

    # Set 999's back to 0
    dilated_sparse_depth = torch.where(
        dilated_sparse_depth == 999,
        torch.zeros_like(sparse_depth),
        dilated_sparse_depth)

    # Create a validity map for dilated depth
    dilated_validity_map = torch.where(dilated_sparse_depth > 0, 1, 0)

    return dilated_sparse_depth, dilated_validity_map

def sparse_depth_error_weight(sparse_depth, dilation_kernel_size, teacher_outputs, w_sparse_error=1):
    '''
    Given sparse depth, return tensor or integer evaluating teacher predictions against sparse depth

    Arg(s):
        sparse_depth : torch.Tensor[float32]
            N x 1 x H x W sparse depth tensor
        dilation_kernel_size : int
            Odd number of how large of a window to repeat sparse depth values
        teacher_outputs : torch.Tensor[float32]
            N x M x H x W teacher output from ensemble of M teachers for left image
        w_sparse_error : float
            weight of sparse depth error. Default value is 1
    Returns:
        torch.Tensor : N x M x H x W tensor evaluating each teacher compared to dilated sparse depth
    '''

    # Obtain dilated sparse depth
    dilated_sparse_depth, dilated_validity_map = dilate_sparse_depth(sparse_depth, dilation_kernel_size)

    # Compare teacher predictions to dilated sparse depth
    _, M, _, _ = teacher_outputs.shape
    dilated_sparse_depth = dilated_sparse_depth.repeat(1, M, 1, 1)
    dilated_validity_map = dilated_validity_map.repeat(1, M, 1, 1)

    # Mask teacher predictions with validity map
    valid_teacher_outputs = torch.where(
        dilated_validity_map > 0,
        teacher_outputs,
        torch.zeros_like(teacher_outputs))

    # Find error between each teacher and the dilated sparse depth
    error = torch.abs(valid_teacher_outputs - dilated_sparse_depth)
    error[dilated_validity_map == 0] = 0

    # Normalize error
    error[dilated_validity_map > 0] /= dilated_sparse_depth[dilated_validity_map > 0]

    # If is_global, then take sum of errors
    ones = torch.ones_like(error)
    error = ones * torch.mean(error, dim=[2, 3], keepdim=True)

    return 1 - torch.exp(-w_sparse_error * error)
