from yacs.config import CfgNode

arch = CfgNode({
    
    'img_shape': (128, 128),
    
    # Grid size. There will be G*G slots
    'G': 8,
    
    # Foreground configurations
    # ==== START ====
    # Foreground likelihood sigma
    'fg_sigma': 0.15,
    # Size of the glimpse
    'glimpse_size': 32,
    # Encoded image feature channels
    'img_enc_dim_fg': 128,
    # Latent dimensions
    'z_pres_dim': 1,
    'z_depth_dim': 1,
    # (h, w)
    'z_where_scale_dim': 2,
    # (x, y)
    'z_where_shift_dim': 2,
    'z_what_dim': 32,
    
    # z_pres prior
    'z_pres_start_step': 4000,
    'z_pres_end_step': 10000,
    'z_pres_start_value': 0.1,
    'z_pres_end_value': 0.01,
    
    # z_scale prior
    'z_scale_mean_start_step': 10000,
    'z_scale_mean_end_step': 20000,
    'z_scale_mean_start_value': -1.0,
    'z_scale_mean_end_value': -2.0,
    'z_scale_std_value': 0.1,
    
    # Temperature for gumbel-softmax
    'tau_start_step': 0,
    'tau_end_step': 10000,
    'tau_start_value': 2.5,
    'tau_end_value': 2.5,
    
    # Turn on boundary loss or not
    'boundary_loss': True,
    # When to turn off boundary loss
    'bl_off_step': 100000000,
    
    # Fix alpha for the first N steps
    'fix_alpha_steps': 0,
    'fix_alpha_value': 0.1,
    # ==== END ====
    
    
    # Background configurations
    # ==== START ====
    # Number of background components. If you set this to one, you should use a strong decoder instead.
    'K': 5,
    # Background likelihood sigma
    'bg_sigma': 0.15,
    # Image encoding dimension
    'img_enc_dim_bg': 64,
    # Latent dimensions
    'z_mask_dim': 32,
    'z_comp_dim': 32,
    
    # (H, W)
    'rnn_mask_hidden_dim': 64,
    # This should be same as above
    'rnn_mask_prior_hidden_dim': 64,
    # Hidden layer dim for the network that computes q(z_c|z_m, x)
    'predict_comp_hidden_dim': 64,
    # ==== END ====
})
