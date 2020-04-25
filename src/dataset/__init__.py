from .atari import Atari
from .obj3d import Obj3D
from torch.utils.data import DataLoader


__all__ = ['get_dataset', 'get_dataloader']

def get_dataset(cfg, mode):
    assert mode in ['train', 'val', 'test']
    if cfg.dataset == 'ATARI':
        mode = 'validation' if mode == 'val' else mode
        return Atari(cfg.dataset_roots.ATARI, mode, gamelist=cfg.gamelist)
    elif cfg.dataset == 'OBJ3D_SMALL':
        return Obj3D(cfg.dataset_roots.OBJ3D_SMALL, mode)
    elif cfg.dataset == 'OBJ3D_LARGE':
        return Obj3D(cfg.dataset_roots.OBJ3D_LARGE, mode)

def get_dataloader(cfg, mode):
    assert mode in ['train', 'val', 'test']
    
    batch_size = getattr(cfg, mode).batch_size
    shuffle = True if mode == 'train' else False
    num_workers = getattr(cfg, mode).num_workers
    
    dataset = get_dataset(cfg, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader
    
