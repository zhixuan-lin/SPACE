__all__ = ['get_evaluator']

from .space_eval import SpaceEval

def get_evaluator(cfg):
    if cfg.model == 'SPACE':
        return SpaceEval()
    
    return None


