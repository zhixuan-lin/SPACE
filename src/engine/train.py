from model import get_model
from eval import get_evaluator
from dataset import get_dataset, get_dataloader
from solver import get_optimizers
from utils import Checkpointer, MetricLogger
import os
import os.path as osp
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from vis import get_vislogger
import time
from torch.nn.utils import clip_grad_norm_



def train(cfg):
    
    print('Experiment name:', cfg.exp_name)
    print('Dataset:', cfg.dataset)
    print('Model name:', cfg.model)
    print('Resume:', cfg.resume)
    if cfg.resume:
        print('Checkpoint:', cfg.resume_ckpt if cfg.resume_ckpt else 'last checkpoint')
    print('Using device:', cfg.device)
    if 'cuda' in cfg.device:
        print('Using parallel:', cfg.parallel)
    if cfg.parallel:
        print('Device ids:', cfg.device_ids)
    
    print('Loading data')

    trainloader = get_dataloader(cfg, 'train')
    if cfg.train.eval_on:
        valset = get_dataset(cfg, 'val')
        # valloader = get_dataloader(cfg, 'val')
        evaluator = get_evaluator(cfg)
    model = get_model(cfg)
    model = model.to(cfg.device)
    checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
    model.train()

    optimizer_fg, optimizer_bg = get_optimizers(cfg, model)

    start_epoch = 0
    start_iter = 0
    global_step = 0
    if cfg.resume:
        checkpoint = checkpointer.load_last(cfg.resume_ckpt, model, optimizer_fg, optimizer_bg)
        if checkpoint:
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step'] + 1
    if cfg.parallel:
        model = nn.DataParallel(model, device_ids=cfg.device_ids)
    
    writer = SummaryWriter(log_dir=os.path.join(cfg.logdir, cfg.exp_name), flush_secs=30, purge_step=global_step)
    vis_logger = get_vislogger(cfg)
    metric_logger =  MetricLogger()

    print('Start training')
    end_flag = False
    for epoch in range(start_epoch, cfg.train.max_epochs):
        if end_flag:
            break
    
        start = time.perf_counter()
        for i, data in enumerate(trainloader):
        
            end = time.perf_counter()
            data_time = end - start
            start = end
        
            model.train()
            imgs = data
            imgs = imgs.to(cfg.device)
            loss, log = model(imgs, global_step)
            # In case of using DataParallel
            loss = loss.mean()
            optimizer_fg.zero_grad()
            optimizer_bg.zero_grad()
            loss.backward()
            if cfg.train.clip_norm:
                clip_grad_norm_(model.parameters(), cfg.train.clip_norm)
        
            optimizer_fg.step()
        
            # if cfg.train.stop_bg == -1 or global_step < cfg.train.stop_bg:
            optimizer_bg.step()
        
            end = time.perf_counter()
            batch_time = end - start
        
            metric_logger.update(data_time=data_time)
            metric_logger.update(batch_time=batch_time)
            metric_logger.update(loss=loss.item())
        
            if (global_step) % cfg.train.print_every == 0:
                start = time.perf_counter()
                log.update({
                    'loss': metric_logger['loss'].median,
                })
                vis_logger.train_vis(writer, log, global_step, 'train')
                end = time.perf_counter()
            
                print(
                    'exp: {}, epoch: {}, iter: {}/{}, global_step: {}, loss: {:.2f}, batch time: {:.4f}s, data time: {:.4f}s, log time: {:.4f}s'.format(
                        cfg.exp_name, epoch + 1, i + 1, len(trainloader), global_step, metric_logger['loss'].median,
                        metric_logger['batch_time'].avg, metric_logger['data_time'].avg, end - start))
                
            if (global_step) % cfg.train.save_every == 0:
                start = time.perf_counter()
                checkpointer.save_last(model, optimizer_fg, optimizer_bg, epoch, global_step)
                print('Saving checkpoint takes {:.4f}s.'.format(time.perf_counter() - start))
        
            if (global_step) % cfg.train.eval_every == 0 and cfg.train.eval_on:
                print('Validating...')
                start = time.perf_counter()
                checkpoint = [model, optimizer_fg, optimizer_bg, epoch, global_step]
                evaluator.train_eval(model, valset, valset.bb_path, writer, global_step, cfg.device, checkpoint, checkpointer)
                print('Validation takes {:.4f}s.'.format(time.perf_counter() - start))
        
            start = time.perf_counter()
            global_step += 1
            if global_step > cfg.train.max_steps:
                end_flag = True
                break





