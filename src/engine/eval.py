from model import get_model
from eval import get_evaluator
from dataset import get_dataset, get_dataloader
from utils import Checkpointer
import os
import os.path as osp
from torch import nn

def eval(cfg):
    assert cfg.resume
    assert cfg.eval.checkpoint in ['best', 'last']
    assert cfg.eval.metric in ['ap_dot5', 'ap_avg']
    
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
    testset = get_dataset(cfg, 'test')
    model = get_model(cfg)
    model = model.to(cfg.device)
    checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
    evaluator = get_evaluator(cfg)
    model.eval()

    use_cpu = 'cpu' in cfg.device
    if cfg.resume_ckpt:
        checkpoint = checkpointer.load(cfg.resume_ckpt, model, None, None, use_cpu)
    elif cfg.eval.checkpoint == 'last':
        checkpoint = checkpointer.load_last('', model, None, None, use_cpu)
    elif cfg.eval.checkpoint == 'best':
        checkpoint = checkpointer.load_best(cfg.eval.metric, model, None, None, use_cpu)
    if cfg.parallel:
        assert 'cpu' not in cfg.device
        model = nn.DataParallel(model, device_ids=cfg.device_ids)
        
    evaldir = osp.join(cfg.evaldir, cfg.exp_name)
    info = {
        'exp_name': cfg.exp_name
    }
    evaluator.test_eval(model, testset, testset.bb_path, cfg.device, evaldir, info)
        
    
