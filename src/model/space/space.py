import torch
import torch.nn as nn
import torch.nn.functional as F
from attrdict import AttrDict
from .arch import arch
from .fg import SpaceFg
from .bg import SpaceBg


class Space(nn.Module):
    
    def __init__(self):
        nn.Module.__init__(self)
        
        self.fg_module = SpaceFg()
        self.bg_module = SpaceBg()
        
    def forward(self, x, global_step):
        """
        Inference.
        
        :param x: (B, 3, H, W)
        :param global_step: global training step
        :return:
            loss: a scalor. Note it will be better to return (B,)
            log: a dictionary for visualization
        """
        
        # Background extraction
        # (B, 3, H, W), (B, 3, H, W), (B,)
        bg_likelihood, bg, kl_bg, log_bg = self.bg_module(x, global_step)
        
        # Foreground extraction
        fg_likelihood, fg, alpha_map, kl_fg, loss_boundary, log_fg = self.fg_module(x, global_step)

        # Fix alpha trick
        if global_step and global_step < arch.fix_alpha_steps:
            alpha_map = torch.full_like(alpha_map, arch.fix_alpha_value)
            
        # Compute final mixture likelhood
        # (B, 3, H, W)
        fg_likelihood = (fg_likelihood + (alpha_map + 1e-5).log())
        bg_likelihood = (bg_likelihood + (1 - alpha_map + 1e-5).log())
        # (B, 2, 3, H, W)
        log_like = torch.stack((fg_likelihood, bg_likelihood), dim=1)
        # (B, 3, H, W)
        log_like = torch.logsumexp(log_like, dim=1)
        # (B,)
        log_like = log_like.flatten(start_dim=1).sum(1)

        # Take mean as reconstruction
        y = alpha_map * fg + (1.0 - alpha_map) * bg
        
        # Elbo
        elbo = log_like - kl_bg - kl_fg
        
        # Mean over batch
        loss = (-elbo + loss_boundary).mean()
        
        log = {
            'imgs': x,
            'y': y,
            # (B,)
            'mse': ((y-x)**2).flatten(start_dim=1).sum(dim=1),
            'log_like': log_like
        }
        log.update(log_fg)
        log.update(log_bg)
        
        return loss, log
