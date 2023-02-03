from cgitb import handler
from pickletools import read_uint1
from turtle import position
import yaml
import sys
import numpy as np
import torch
import time
import random
import shutil
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, TypedDict
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Dict
from collections import OrderedDict, defaultdict, deque
from omegaconf import OmegaConf




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def yaml_load(filename):
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    return data

def yaml_save(filename, dict):
    with open(filename, 'w') as f:
        yaml.dump(dict, f, default_flow_style=False)

def dict_of_dicts_merge(x, y):
    z = {}
    overlapping_keys = x.keys() & y.keys()
    for key in overlapping_keys:
        if isinstance(y[key], dict) and isinstance(x[key], dict):
            z[key] = dict_of_dicts_merge(x[key], y[key])
        else:
            z[key] = y[key]
    for key in x.keys() - overlapping_keys:
        z[key] = deepcopy(x[key])
    for key in y.keys() - overlapping_keys:
        z[key] = deepcopy(y[key])
    return z



def get_model_device(model):
    return next(model.parameters()).device

def get_device(gpu_index):
    if gpu_index >= 0:
        try:
            assert torch.cuda.is_available(), 'cuda not available'
            assert gpu_index >= 0 and gpu_index < torch.cuda.device_count(), 'gpu index out of range'
            return torch.device('cuda:{}'.format(gpu_index))
        except AssertionError:
            return torch.device('cpu')
    else:
        return torch.device('cpu')

def get_optimizer(model, optim, lr, l2):
    if optim == 'adam':
        return torch.optim.Adam( list(filter(lambda p: p.requires_grad, model.parameters())), lr=lr, weight_decay=l2)
    elif optim == 'sgd':
        return torch.optim.SGD( list(filter(lambda p: p.requires_grad, model.parameters())), lr=lr, weight_decay=l2)
    else:
        raise NotImplementedError

def get_scheduler(optimizer, step_size, gamma, last_epoch):
    from torch.optim.lr_scheduler import StepLR    
    scheduler = StepLR(optimizer, step_size, gamma, last_epoch)
    return scheduler



# another get_logger function
def get_default_logger(args={}, log_dir='./', exp_name=None, to_file=True, to_console=True):
    log_dir = Path(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    logger.handlers = []
    if to_console:
        sh = logging.StreamHandler(stream=sys.stdout) # add command line stream handler
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    if to_file: # add file handler
        file_path = log_dir / f"{exp_name}.log"
        fh = logging.FileHandler(file_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info('Create log file at {}'.format(file_path))

    
    return logger

def get_default_recorder():
    return None


def split_train_val_test(itemlist, vt_ratio=0.3):
    """
    Split a list to train/val/test sets.

    Args:
        itemlist: a list of items for splitting.
        vt_ratio: ratio of val+test.
    """
    index = np.random.permutation(len(itemlist))
    train_end = int(len(itemlist) * (1-vt_ratio))
    val_end = train_end + int(len(itemlist) * vt_ratio/2)
    train_idx = index[:train_end]
    val_idx = index[train_end:val_end]
    test_idx = index[val_end:]

    train_list = []
    val_list = []
    test_list = []
    for idx in train_idx:
        train_list.append(itemlist[idx])
    for idx in val_idx:
        val_list.append(itemlist[idx])
    for idx in test_idx:
        test_list.append(itemlist[idx])
    
    return train_list, val_list, test_list



def save_fig(filename, fig):
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def ax_plot(ax, data, label):
    ax.plot(data, label=label)

def set_ax(ax, metric='metric'):
    ax.set_title(metric)
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric)
    ax.legend(loc='upper right')
    pass

def makedirs(dir):
    if not isinstance(dir, Path):
        dir = Path(dir)
    if not dir.exists():
        os.makedirs(dir)
    return dir

class Saver():
    def __init__(self, checkpoint_dir, frequency, val_key=None, val_higher_better=True):
        self.checkpoint_dir = checkpoint_dir
        self.frequency = frequency
        self.val_higher_better = val_higher_better
        if self.val_higher_better:
            self.val_curr_best = float('-inf')
        else:
            self.val_curr_best = float('inf')

    def save_model(self, model, epoch_i, val_results=None, val_key=None):
        ensure_dir_exists(self.checkpoint_dir)

        
        if (epoch_i + 1) % self.frequency == 0:
            ckpt_name = self.checkpoint_dir/f'epoch_{epoch_i}.state_dict'
            torch.save( model.state_dict(), ckpt_name)
            print(f'Epoch {epoch_i} ckpt saved.')

    @staticmethod
    def load_model(ckpt_file, map_location=None):
        
        state_dict = torch.load(ckpt_file, map_location=map_location)
        return state_dict

def ensure_dir_exists(dir):
    dir = Path(dir)
    if not dir.exists():
        os.makedirs(dir)


class DataLoaders(object):
    def __init__(self, dataloaders):
        self.train_dataloader = dataloaders['train']
        self.val_dataloader = dataloaders['val']
        self.test_dataloader = dataloaders['test']

        self.curr_dataloader = self.train_dataloader # default using train_dataloader.

    def __len__(self):
        return len(self.curr_dataloader)

    def train(self):
        """Set to train mode(dataset)"""
        self.curr_dataloader = self.train_dataloader

    def val(self):
        """Set to val mode(dataset)"""
        self.curr_dataloader = self.val_dataloader

    def test(self):
        """Set to test mode(dataset)"""
        self.curr_dataloader = self.test_dataloader



class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def set_device(self, device):
        self = self.to(device)
    
    def forward_pass(self, batch):
        """Process one batch
        Args:
            batch: a batch of data. Labels are contained in batch.

        Return:
            {loss: torch.tensor, key: torch.tensor, ... }
        """

        raise NotImplementedError
    


def process_results(results_list):
    """
    Args:
        results_list: [
            {k1: val1_k1, k2: val1_k2}, # batch 1
            {k1: val2_k2, k2: val2_k2}, # batch 2
            ...
            ]
    Return:
        results_dict: {
            k1: [val1_k1, val2_k2, ...],
            k2: [val1_k2, val2_k2, ...],
        }
    """
    # detach torch values to scalars
    for i, dic in enumerate(results_list):
        dic_detach = {}
        for k, v in dic.items():
            if isinstance(v, torch.Tensor):
                if v.dim() == 0: # scalar
                    dic_detach[k] = v.detach().cpu().item()
                else: # tensor
                    dic_detach[k] = v.detach().cpu().numpy()
                    # dic_detach[k] = v.detach()
            else: 
                dic_detach[k] = v
            
        results_list[i] = dic_detach
    
    # aggregate each metric
    results_dic = defaultdict(list)
    for i, dic in enumerate(results_list):
        for k, v in dic.items():
            results_dic[k].append( dic[k] )
    results_dic = dict(results_dic)


    return results_dic

def str_dict(dict):
    dict_str = ''
    items = list(dict.items())
    for i, (key, value) in enumerate(items):
        if isinstance(value, int) or isinstance(value, np.int64):
            dict_str = dict_str + str(key) + ':' + f' {value:d}'
        else:
            dict_str = dict_str + str(key) + ':' + f' {value:.12f}'
        if i != len(items) - 1:
            dict_str = dict_str + ', '
    string = '{' + f"{dict_str}" + '}'
    return string

class EarlyStopper:
    def __init__(self, metric, larger_better=True, tolerance=10) -> None:
        self.metric = metric
        self.larger_better = larger_better
        self.best_value = float('-inf') if larger_better else float('inf')
        self.best_epoch = -1
        self.tolerance_base = tolerance
        self.tolerance_curr = tolerance
    
    def stop(self, validation_result, epoch):
        value = np.mean( validation_result[self.metric] )

        if (self.larger_better and value > self.best_value) or (not self.larger_better and value < self.best_value):
            # if value > self.best_value:
                self.best_epoch = epoch
                self.best_value = value
                self.tolerance_curr = self.tolerance_base
        else:
            self.tolerance_curr -= 1
        
        if self.tolerance_curr <= 0:
            return True
        else: return False



class Trainer(object):
    def __init__(self, 
                    model_name,
                    dataset_name,
                    model_constructor,
                    model_constructor_params,
                    data_loaders: DataLoaders, 
                    optimizer_constructor,
                    optimizer_params,
                    scheduler_constructor,
                    scheduler_params,
                    random_seed: int,
                    epochs, 
                    device, 
                    early_stoper: EarlyStopper=None,
                    logger=None,
                    tb_log_dir=None,
                    log2tensorboard=False,
                    hparams_save: Dict=None, # hyper-parameters for one experiment.
                    hparams_save_dir=None, # hyper-parameters save dir for one experiment.
                    ckpt_dir=None,
                    log_dir=None,
                    tmp_dir=None,
                    debug=True,
                    experiment_name=None,
                    batch_filter=None,
                    training_mode=True,
                    train_epoch_callbacks=[], # run after each train epoch
                    **kwargs):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.model_constructor = model_constructor
        self.model_constructor_params = model_constructor_params
        self.early_stoper = early_stoper

        self.data_loader = data_loaders
        self.epochs = epochs
        self.device = device
        self.random_seed = random_seed
        self.debug = debug 

        self.optimizer_constructor = optimizer_constructor
        self.optimizer_params = optimizer_params
        self.scheduler_constructor = scheduler_constructor
        self.scheduler_params = scheduler_params

        self.batch_filter = batch_filter

        self.logger = logger 
        self.log_dir = log_dir
        self.tmp_dir = tmp_dir
        
        self.pid = os.getpid()
        
       
        self.ckpt_dir = ckpt_dir
        self.model_saver = Saver(ckpt_dir/experiment_name, frequency=10)

        
        self.tb_log_dir = tb_log_dir
        self.tb_writer = None 
        self.hparams_save = hparams_save
        self.hparams_save_dir = hparams_save_dir
        self.log2tensorboard = log2tensorboard

        self.training_mode = training_mode
        self.EXP_NAME = experiment_name


        self.kwargs = kwargs
        self.val = self.kwargs.get('val', False) 
        


        self.train_sample_num = len(self.data_loader.train_dataloader.dataset)
        self.test_sample_num = len(self.data_loader.test_dataloader.dataset)

        self.train_epoch_callbacks = train_epoch_callbacks

        
        self.__init_model__()
    
    def __init_model__(self,):
        if self.training_mode:
            self._save_hparams()
        
        if self.training_mode and not self.debug:
            self._append_exp_name()

            if self.log2tensorboard:
                self.tb_writer = SummaryWriter(log_dir=self.tb_log_dir/self.EXP_NAME) # only create a log folder when model finished one epoch. To avoid redundant folders.

        
        print(f'\nrandom seed: {self.random_seed}')
        set_seed(self.random_seed)

        self.model = self.model_constructor(**self.model_constructor_params)
        self.model.__trainer__ = self # to let the model could access some trainer information, e.g., current epoch
        self.optimizer = self.model.configure_optimizers()
        self.scheduler = None
        self.cur_epoch = -1

        

    @property
    def hparams_path(self,):
        return self.hparams_save_dir/f'{self.EXP_NAME}.yml'

    def _save_hparams(self, ):
        with open(self.hparams_path, 'w') as f:
            f.write( OmegaConf.to_yaml(self.hparams_save) )
    
    def _append_exp_name(self,):
        ALL_EXP_RECORDS = self.log_dir/f"ALL_EXP_RECORDS_{self.dataset_name}_{self.model_name}.txt"
        with open(ALL_EXP_RECORDS, 'a+') as f:
            f.write(self.EXP_NAME + ',' + '\n')
        
        ALL_EXP_RECORDS = self.log_dir/f"ALL_EXP_RECORDS.txt"
        with open(ALL_EXP_RECORDS, 'a+') as f:
            f.write(self.EXP_NAME + ',' + '\n')
        
    def _save_current_state(self):
        task_id = self.kwargs.get('task_id', -1) # support for multi-task
        if task_id != -1:
            state_dict = {
                'mode': self.mode,
                'total_epoch': self.epochs,
                'cur_epoch': self.cur_epoch,
            }
            with open(self.tmp_dir/f'task{task_id}.yml', 'w') as f:
                yaml.dump(state_dict, f)
        else:
            return
    

    def step_compute(self, batch, batch_idx): # batch compute
        if self.batch_filter is not None and self.batch_filter(batch):
            return None
        
        if self.mode == 'train':
            try:
                step_results = self.model.forward_pass(batch, batch_idx)
                if getattr(self.model, 'backward', None) is not None and callable(getattr(self.model, 'backward')):
                    self.model.backward(self.optimizer, step_results)
                else:
                    self.optimizer.zero_grad()
                    step_results['loss'].backward()
                    self.optimizer.step()
            except Exception as e:
                raise
                if 'CUDA out of memory' in e.args[0]:
                    # print(e.args[0])
                    raise
                    # exit(1)
                else: raise
        else:
            with torch.no_grad():
                step_results = self.model.forward_pass(batch, batch_idx)

        return step_results


    def process_all_batchs(self, desc):
        num_batches = len(self.data_loader.curr_dataloader)

        all_batch_results = []
        desc = desc + f"[pid {self.pid}]"
        if self.kwargs.get('task_id', -1) == -1:
            all_batchs = tqdm(enumerate(self.data_loader.curr_dataloader), total=num_batches, desc=desc, leave=True, colour='green')
        else:
            all_batchs = enumerate(self.data_loader.curr_dataloader)
        self.total_batchs = len(self.data_loader.curr_dataloader)
        
        for batch_idx, batch in all_batchs:
            step_results = self.step_compute(batch, batch_idx)
            self.cur_batch = batch_idx

            if step_results is not None:
                all_batch_results.append(step_results)

                
        self._save_current_state()

        all_batch_results = process_results(all_batch_results)

        return all_batch_results


    def train_epoch(self):
        """ train one epoch """
        self.model.train()
        self.data_loader.train()
        self.mode = 'train'
        tqdm_desc = f'[train][epoch {self.cur_epoch+1}/{self.epochs}]'
        results_dic = self.process_all_batchs(tqdm_desc)
        return results_dic



    def val_epoch(self):
        self.model.eval()
        self.data_loader.val()
        self.mode = 'val'
        tqdm_desc = f'[val  ][epoch {self.cur_epoch+1}/{self.epochs}]'
        results_dic = self.process_all_batchs(tqdm_desc)

        return results_dic


    def test_epoch(self):
        self.model.eval()
        self.data_loader.test()
        self.mode = 'test'
        tqdm_desc = f'[test ][epoch {self.cur_epoch+1}/{self.epochs}]'
        results_dic = self.process_all_batchs(tqdm_desc)

        return results_dic

    def train(self):
        """ Train model """

        # one run
        train_times = []
        test_times = []
        for i in range(self.epochs):
            self.cur_epoch = i
            
            # train
            start_time = time.time()
            train_results_epoch = self.train_epoch()
            train_times.append(time.time() - start_time)

            # val
            if self.val:
                val_results = self.val_epoch()
            else:
                val_results = None
            
            # test
            start_time = time.time()
            test_results_epoch = self.test_epoch()
            test_times.append(time.time() - start_time)
            
            for callback in self.train_epoch_callbacks:
                callback(train_results=train_results_epoch,
                        val_results=val_results,
                        test_results=test_results_epoch,
                        cur_epoch=self.cur_epoch,
                        logger=self.logger,
                        tb_writer=self.tb_writer,
                        model=self.model,
                        __trainer__=self,
                        )
            
            if self.scheduler is not None:
                self.scheduler.step()

            if self.early_stoper is not None:
                if self.early_stoper.stop(val_results, self.cur_epoch): 
                    self.model_saver.save_model(self.model, self.cur_epoch, val_results=None, val_key=None)
                    self.logger.info(f'Early stopped at epoch {self.cur_epoch}. Saved')
                    break

            assert self.model_saver is not None
            self.model_saver.save_model(self.model, self.cur_epoch, val_results=None, val_key=None)
            print('\n')
        
        train_time_per_sample = np.mean(train_times)/self.train_sample_num*1000
        test_time_per_sample = np.mean(test_times)/self.test_sample_num*1000
        print(f'Train time: {train_time_per_sample:.6f}ms, test time: {test_time_per_sample:.6f}ms (per sample)\n')

    def evaluate(self, ckpt_file, evaluate_data=False, evaluation_callbacks=[]):
        print(f'ckpt loaded: {ckpt_file}')
        state_dict = Saver.load_model(ckpt_file, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.debug = True
        if evaluate_data:
            test_results = self.test_epoch()
            assert self.model.training is False
        else:
            test_results = None
        
        for callback in evaluation_callbacks:
            callback(
                test_results=test_results,
                model=self.model,
                dataloaders=self.data_loader,
                experiment_name=self.EXP_NAME,
                __trainer__=self,
            )
        pass





    
