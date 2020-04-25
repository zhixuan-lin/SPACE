from yacs.config import CfgNode
eval_cfg = CfgNode({
    # Evaluation during training
    'train': {
        # What to evaluate
        'metrics': ['mse', 'ap'],
        # Number of samples for evaluation
        'num_samples': {
            'mse': 200,
            'ap': 200,
        },
        
        # For dataloader
        'batch_size': 12,
        'num_workers': 4,
    },
    'test': {
        # For dataloader
        'batch_size': 12,
        'num_workers': 4,
    }
    
})
