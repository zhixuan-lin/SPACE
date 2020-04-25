from model import get_model
from vis import get_vislogger
from dataset import get_dataset, get_dataloader
from utils import Checkpointer
import os
import os.path as osp
from torch import nn


def show(cfg):
    assert cfg.resume, 'You must pass "resume True" to the if --task is "show"'
    
    print('Experiment name:', cfg.exp_name)
    print('Dataset:', cfg.dataset)
    print('Model name:', cfg.model)
    print('Resume:', cfg.resume)
    if cfg.resume:
        print('Checkpoint:', cfg.resume_ckpt if cfg.resume_ckpt else 'see below')
    print('Using device:', cfg.device)
    if 'cuda' in cfg.device:
        print('Using parallel:', cfg.parallel)
    if cfg.parallel:
        print('Device ids:', cfg.device_ids)
    
    print('Loading data')
    dataset = get_dataset(cfg, cfg.show.mode)
    model = get_model(cfg)
    model = model.to(cfg.device)
    checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
    vis_logger = get_vislogger(cfg)
    model.eval()
    
    use_cpu = 'cpu' in cfg.device
    if cfg.resume_ckpt:
        checkpoint = checkpointer.load(cfg.resume_ckpt, model, None, None, use_cpu=use_cpu)
    else:
        # Load last checkpoint
        checkpoint = checkpointer.load_last('', model, None, None, use_cpu=use_cpu)
        
    if cfg.parallel:
        assert 'cpu' not in cfg.device, 'You can not set "parallel" to True when you set "device" to cpu'
        model = nn.DataParallel(model, device_ids=cfg.device_ids)
    
    os.makedirs(cfg.demodir, exist_ok=True)
    img_path = osp.join(cfg.demodir, '{}.png'.format(cfg.exp_name))
    vis_logger.show_vis(model, dataset, cfg.show.indices, img_path, device=cfg.device)
    print('The result image has been saved to {}'.format(osp.abspath(img_path)))
    


