import torch
import torch.nn.functional as F
from torch.distributions import RelaxedBernoulli
from torch.distributions.utils import broadcast_all


def spatial_transform(image, z_where, out_dims, inverse=False):
    """ spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True
    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]
    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T
    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    """
    # 1. construct 2x3 affine matrix for each datapoint in the minibatch
    theta = torch.zeros(2, 3).repeat(image.shape[0], 1, 1).to(image.device)
    # set scaling
    theta[:, 0, 0] = z_where[:, 0] if not inverse else 1 / (z_where[:, 0] + 1e-9)
    theta[:, 1, 1] = z_where[:, 1] if not inverse else 1 / (z_where[:, 1] + 1e-9)

    # set translation
    theta[:, 0, -1] = z_where[:, 2] if not inverse else - z_where[:, 2] / (z_where[:, 0] + 1e-9)
    theta[:, 1, -1] = z_where[:, 3] if not inverse else - z_where[:, 3] / (z_where[:, 1] + 1e-9)
    # 2. construct sampling grid
    grid = F.affine_grid(theta, torch.Size(out_dims))
    # 3. sample image from grid
    return F.grid_sample(image, grid)

def linear_annealing(device, step, start_step, end_step, start_value, end_value):
    """
    Linear annealing
    
    :param x: original value. Only for getting device
    :param step: current global step
    :param start_step: when to start changing value
    :param end_step: when to stop changing value
    :param start_value: initial value
    :param end_value: final value
    :return:
    """
    if step <= start_step:
        x = torch.tensor(start_value, device=device)
    elif start_step < step < end_step:
        slope = (end_value - start_value) / (end_step - start_step)
        x = torch.tensor(start_value + slope * (step - start_step), device=device)
    else:
        x = torch.tensor(end_value, device=device)
    
    return x


class NumericalRelaxedBernoulli(RelaxedBernoulli):
    """
    This is a bit weird. In essence it is just RelaxedBernoulli with logit as input.
    """
    
    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(sample_shape)
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        diff = logits - value.mul(self.temperature)
        
        out = self.temperature.log() + diff - 2 * diff.exp().log1p()
        
        return out


def kl_divergence_bern_bern(z_pres_logits, prior_pres_prob, eps=1e-15):
    """
    Compute kl divergence of two Bernoulli distributions
    :param z_pres_logits: (B, ...)
    :param prior_pres_prob: float
    :return: kl divergence, (B, ...)
    """
    z_pres_probs = torch.sigmoid(z_pres_logits)
    kl = z_pres_probs * (torch.log(z_pres_probs + eps) - torch.log(prior_pres_prob + eps)) + \
         (1 - z_pres_probs) * (torch.log(1 - z_pres_probs + eps) - torch.log(1 - prior_pres_prob + eps))
    
    return kl


def get_boundary_kernel_new(kernel_size=32, boundary_width=6):
    """
    Will return something like this:
    ============
    =          =
    =          =
    ============
    size will be (kernel_size, kernel_size)
    """
    filter = torch.zeros(kernel_size, kernel_size)
    filter[:, :] = 1.0 / (kernel_size ** 2)
    # Set center to zero
    filter[boundary_width:kernel_size - boundary_width, boundary_width:kernel_size - boundary_width] = 0.0
    
    return filter


def get_boundary_kernel(kernel_size=32, sigma=20, channels=1, beta=1.0):
    """
    TODO: This function is no longer used.
    """
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    boundary_kernel = x_coord.repeat(kernel_size).view(kernel_size, kernel_size).float()
    
    part_sum = 1.0
    # boundary = int((1 - math.sqrt(2) / 2) * kernel_size / 2)
    boundary = int((kernel_size - sigma) / 2)
    num_center = pow(kernel_size - boundary - boundary, 2)
    num_boundary = kernel_size * kernel_size - num_center
    boundary_kernel.data[:, :] = -part_sum / (num_boundary + num_center) * beta
    boundary_kernel.data[
    boundary: kernel_size - boundary, boundary: kernel_size - boundary
    ] = part_sum / (num_boundary + num_center)
    
    # Reshape to 2d depthwise convolutional weight
    boundary_kernel = boundary_kernel.view(1, 1, kernel_size, kernel_size)
    boundary_kernel = boundary_kernel.repeat(channels, 1, 1, 1)
    
    boundary_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                      kernel_size=kernel_size, groups=channels, bias=False)
    
    boundary_filter.weight.data = boundary_kernel
    boundary_filter.weight.requires_grad = False
    return boundary_filter
