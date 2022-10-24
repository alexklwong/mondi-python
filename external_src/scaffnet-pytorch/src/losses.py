import torch


EPSILON = 1e-8

def l1_loss_func(src, tgt, w, normalize=False):
    '''
    Computes the L1 difference between source and target

    Arg(s):
        src : torch.Tensor[float32]
            source tensor
        tgt : torch.Tensor[float32]
            target tensor
        w : torch.Tensor[float32]
            weights for penalty
    Returns:
        float : mean L1 penalty
    '''
    loss = w * torch.abs(tgt - src)
    if normalize:
        loss = loss / (torch.abs(tgt) + EPSILON)

    return torch.mean(loss)

def l1_with_uncertainty_loss_func(src, tgt, uncertainty, w):
    '''
    Computes the l1 loss with uncertainty between source and target

    Arg(s):
        src : torch.Tensor[float32]
            source tensor
        tgt : torch.Tensor[float32]
            target tensor
        uncertainty : torch.Tensor[float32]
            uncertainty map
        w : torch.Tensor[float32]
            weights for penalty
    Returns:
        float : mean L1 penalty with uncertainty
    '''

    loss = w * ((torch.abs(tgt - src) / torch.exp(uncertainty)) + uncertainty)

    return torch.mean(loss)

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
        (1) [reduce_loss=True] torch.Tensor[float32] : mean 1 - SSIM scores between source and target images
        (2) [reduce_loss=False] torch.Tensor[float32] : N x 1 x H x W tensor of 1 - SSIM scores between source and target images
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

def smoothness_loss_func(predict, image, w=None):
    '''
    Computes the local smoothness loss

    Arg(s):
        predict : torch.Tensor[float32]
            N x 1 x H x W predictions
        image : torch.Tensor[float32]
            N x 3 x H x W RGB image
        w : torch.Tensor[float32]
            N x 1 x H x W weights
    Returns:
        torch.Tensor[float32] : smoothness loss
    '''

    predict_dy, predict_dx = gradient_yx(predict)
    image_dy, image_dx = gradient_yx(image)

    # Create edge awareness weights
    weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    if w is not None:
        weights_x = weights_x * w[:, :, :, :-1]
        weights_y = weights_y * w[:, :, :-1, :]

    smoothness_x = torch.mean(weights_x * torch.abs(predict_dx))
    smoothness_y = torch.mean(weights_y * torch.abs(predict_dy))

    return smoothness_x + smoothness_y

def prior_depth_consistency_loss_func(src, tgt, w, normalize=False):
    '''
    Computes the prior depth consistency loss

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

    delta = torch.abs(tgt - src)
    loss = torch.sum(w * delta, dim=[1, 2, 3])

    return torch.mean(loss / torch.sum(w, dim=[1, 2, 3]))


'''
Helper functions for constructing loss functions
'''
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
