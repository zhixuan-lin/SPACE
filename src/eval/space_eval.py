from utils import MetricLogger
import numpy as np
import torch
import math
import os, sys
from datetime import datetime
from torch.utils.data import Subset, DataLoader
import os.path as osp
from tqdm import tqdm
import json
from .eval_cfg import eval_cfg
from .ap import read_boxes, convert_to_boxes, compute_ap, compute_counts
from torch.utils.tensorboard import SummaryWriter


class SpaceEval():
    def __init__(self):
        pass

    @torch.no_grad()
    def test_eval(self, model, testset, bb_path, device, evaldir, info):
        result_dict = self.eval_ap_and_acc(
            model, testset, bb_path, eval_cfg.test.batch_size, eval_cfg.test.num_workers,
            device, num_samples=None
        )
        os.makedirs(evaldir, exist_ok=True)
        path = osp.join(evaldir, 'results_{}.json'.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
        self.save_to_json(result_dict, path, info)
        self.print_result(result_dict, [sys.stdout, open('./results.txt', 'w')])
        # APs = result_dict['APs']
        # iou_thresholds = result_dict['iou_thresholds']
        # accuracy = result_dict['accuracy']
        # perfect = result_dict['perfect']
        # overcount = result_dict['O']
        # undercount = result_dict['undercount']
        # error_rate = result_dict['error_rate']
        

    @torch.no_grad()
    def train_eval(self, model, valset, bb_path, writer, global_step, device, checkpoint, checkpointer):
        """
        Evaluation during training. This includes:
            - mse evaluated on validation set
            - ap and accuracy evaluated on validation set
        :return:
        """
        if 'mse' in eval_cfg.train.metrics:
            self.train_eval_mse(model, valset, writer, global_step, device)
        if 'ap' in eval_cfg.train.metrics:
            results = self.train_eval_ap_and_acc(model, valset, bb_path, writer, global_step, device)
            checkpointer.save_best('ap_dot5', results['APs'][0], checkpoint, min_is_better=False)
            checkpointer.save_best('ap_avg', np.mean(results['APs']), checkpoint, min_is_better=False)
            checkpointer.save_best('error_rate', results['error_rate'], checkpoint, min_is_better=True)
        
    @torch.no_grad()
    def train_eval_ap_and_acc(self, model, valset, bb_path, writer: SummaryWriter, global_step, device):
        """
        Evaluate ap and accuracy during training
        
        :return: result_dict
        """
        result_dict = self.eval_ap_and_acc(
            model, valset, bb_path, eval_cfg.train.batch_size, eval_cfg.train.num_workers,
            device, num_samples=eval_cfg.train.num_samples.ap
        )
        APs = result_dict['APs']
        iou_thresholds = result_dict['iou_thresholds']
        accuracy = result_dict['accuracy']
        perfect = result_dict['perfect']
        overcount = result_dict['overcount']
        undercount = result_dict['undercount']
        error_rate = result_dict['error_rate']
        
        for ap, thres in zip(APs, iou_thresholds):
            writer.add_scalar(f'val/ap_{thres}', ap, global_step)
        writer.add_scalar(f'val/ap_avg', np.mean(APs), global_step)
        writer.add_scalar('val/accuracy', accuracy, global_step)
        writer.add_scalar('val/perfect', perfect, global_step)
        writer.add_scalar('val/overcount', overcount, global_step)
        writer.add_scalar('val/undercount', undercount, global_step)
        writer.add_scalar('val/error_rate', error_rate, global_step)
        
        return result_dict

    @torch.no_grad()
    def train_eval_mse(self, model, valset, writer, global_step, device):
        """
        Evaluate MSE during training
        """
        num_samples = eval_cfg.train.num_samples.mse
        batch_size = eval_cfg.train.batch_size
        num_workers = eval_cfg.train.num_workers
        
        model.eval()
        valset = Subset(valset, indices=range(num_samples))
        dataloader = DataLoader(valset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
        metric_logger = MetricLogger()
    
        print(f'Evaluating MSE using {num_samples} samples.')
        with tqdm(total=num_samples) as pbar:
            for batch_idx, sample in enumerate(dataloader):
                imgs = sample.to(device)
                loss, log = model(imgs, global_step)
                B = imgs.size(0)
                for b in range(B):
                    metric_logger.update(
                        mse=log['mse'][b],
                    )
                metric_logger.update(loss=loss.mean())
                pbar.update(B)
    
        assert metric_logger['mse'].count == num_samples
        # Add last log
        # log.update([(k, torch.tensor(v.global_avg)) for k, v in metric_logger.values.items()])
        mse = metric_logger['mse'].global_avg
        writer.add_scalar(f'val/mse', mse, global_step=global_step)
    
        model.train()
        
        return mse

    def eval_ap_and_acc(
            self,
            model,
            dataset,
            bb_path, 
            batch_size,
            num_workers,
            device,
            num_samples=None,
            iou_thresholds=None,
    ):
        """
        Evaluate average precision and accuracy
        
        :param model: Space
        :param dataset: dataset
        :param bb_path: directory containing the gt bounding boxes.
        :param batch_size: batch size
        :param num_workers: num_workers
        :param device: device
        :param output_path: checkpointdir to output result json file
        :param num_samples: number of samples for evaluating it. If None use all samples
        :param iou_thresholds:
        :return ap: a list of average precisions, corresponding to each iou_thresholds
        """
        
        from tqdm import tqdm
        import sys
    
        model.eval()
        
        if num_samples is None:
            num_samples = len(dataset)
        dataset = Subset(dataset, indices=range(num_samples))
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        
        if iou_thresholds is None:
            iou_thresholds = np.linspace(0.5, 0.95, 10)
        boxes_gt = read_boxes(bb_path, 128)
    
        boxes_pred = []
        model.eval()
        with torch.no_grad():
            print('Computing boxes...')
            pbar = tqdm(total=len(dataloader))
            for i, imgs in enumerate(dataloader):
                imgs = imgs.to(device)
            
                # TODO: treat global_step in a more elegant way
                loss, log = \
                    model(imgs, global_step=100000000)
            
                # (B, N, 4), (B, N, 1), (B, N, 1)
                z_where, z_pres_prob = log['z_where'], log['z_pres_prob']
                # (B, N, 4), (B, N), (B, N)
                z_where = z_where.detach().cpu()
                z_pres_prob = z_pres_prob.detach().cpu().squeeze()
                # TODO: look at this
                z_pres = z_pres_prob > 0.5
            
                boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob)
                boxes_pred.extend(boxes_batch)
                pbar.update(1)
        
            print('Computing error rates and counts...')
            # Four numbers
            error_rate, perfect, overcount, undercount = compute_counts(boxes_pred, boxes_gt)
            accuracy = perfect / (perfect + overcount + undercount)
        
            print('Computing average precision...')
            # A list of length 10
            APs = compute_ap(boxes_pred, boxes_gt, iou_thresholds)
    
        model.train()
        
        return {
            'APs': APs,
            'iou_thresholds': iou_thresholds,
            'error_rate': error_rate,
            'perfect': perfect,
            'accuracy': accuracy,
            'overcount': overcount,
            'undercount': undercount
        }
        
        
    def save_to_json(self, result_dict, json_path, info):
        """
        Save evaluation results to json file
        
        :param result_dict: a dictionary
        :param json_path: checkpointdir
        :param info: any other thing you want to save
        :return:
        """
        from collections import OrderedDict
        import json
        from datetime import datetime
        tosave = OrderedDict([
            ('date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ('info', info),
            ('APs', list(result_dict['APs'])),
            ('iou_thresholds', list(result_dict['iou_thresholds'])),
            ('AP average', np.mean(result_dict['APs'])),
            ('error_rate', result_dict['error_rate']),
            ('accuracy', result_dict['accuracy']),
            ('perfect', result_dict['perfect']),
            ('undercount', result_dict['undercount']),
            ('overcount', result_dict['overcount']),
        ])
        with open(json_path, 'w') as f:
            json.dump(tosave, f, indent=2)
        
        print(f'Results have been saved to {json_path}.')
    
    def print_result(self, result_dict, files):
        APs = result_dict['APs']
        iou_thresholds = result_dict['iou_thresholds']
        accuracy = result_dict['accuracy']
        perfect = result_dict['perfect']
        overcount = result_dict['overcount']
        undercount = result_dict['undercount']
        error_rate = result_dict['error_rate']
        for file in files:
            print('-' * 30, file=file)
            print('{:^15} {:^15}'.format('IoU threshold', 'AP'), file=file)
            print('{:15} {:15}'.format('-' * 15, '-' * 15), file=file)
            for thres, ap in zip(iou_thresholds, APs):
                print('{:<15.2} {:<15.4}'.format(thres, ap), file=file)
            print('{:15} {:<15.4}'.format('Average:', np.mean(APs)), file=file)
            print('{:15} {:15}'.format('-' * 15, '-' * 15), file=file)
        
            print('{:15} {:<15}'.format('Perfect:', perfect), file=file)
            print('{:15} {:<15}'.format('Overcount:', overcount), file=file)
            print('{:15} {:<15}'.format('Undercount:', undercount), file=file)
            print('{:15} {:<15.4}'.format('Accuracy:', accuracy), file=file)
            print('{:15} {:<15.4}'.format('Error rate:', error_rate), file=file)
            print('{:15} {:15}'.format('-' * 15, '-' * 15), file=file)
            
    # def save_best(self, evaldir, metric_name, value, checkpoint, checkpointer, min_is_better):
    #     metric_file = os.path.join(evaldir, f'best_{metric_name}.json')
    #     checkpoint_file = os.path.join(evaldir, f'best_{metric_name}.pth')
    #
    #     now = datetime.now()
    #     log = {
    #         'name': metric_name,
    #         'value': float(value),
    #         'date': now.strftime("%Y-%m-%d %H:%M:%S"),
    #         'global_step': checkpoint[-1]
    #     }
    #
    #     if not os.path.exists(metric_file):
    #         dump = True
    #     else:
    #         with open(metric_file, 'r') as f:
    #             previous_best = json.load(f)
    #         if not math.isfinite(log['value']):
    #             dump = True
    #         elif (min_is_better and log['value'] < previous_best['value']) or (
    #                 not min_is_better and log['value'] > previous_best['value']):
    #             dump = True
    #         else:
    #             dump = False
    #     if dump:
    #         with open(metric_file, 'w') as f:
    #             json.dump(log, f)
    #         checkpointer.save(checkpoint_file, *checkpoint)
