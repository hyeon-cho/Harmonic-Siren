import torch 
import os
import shutil
import time
import logging
import threading

import numpy as np
from torch.optim import SGD, Adam, AdamW
from tensorboardX import SummaryWriter

from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

def ensure_path(path, replace=True, force_replace=False):
    is_temp = os.path.basename(path.rstrip('/')).startswith('_')
    if os.path.exists(path):
        if replace and (is_temp or force_replace or input(f'{path} exists, replace? y/[n]') == 'y'):
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.makedirs(path)


def set_logger(file_path):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path, 'a')
    formatter = logging.Formatter('[%(asctime)s] %(message)s', '%m-%d %H:%M:%S')
    for handler in [stream_handler, file_handler]:
        handler.setFormatter(formatter)
        handler.setLevel('INFO')
        logger.addHandler(handler)
    return logger


def set_save_dir(save_dir, replace=True):
    ensure_path(save_dir, replace=replace)
    logger = set_logger(os.path.join(save_dir, 'log.txt'))
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))
    return logger, writer


def compute_num_params_all(model, text=True):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            s = '{:.1f}M'.format(tot / 1e6)
        else:
            s = '{:.1f}K'.format(tot / 1e3)
        return f'{s} ({tot})'
    else:
        return tot


def get_model_param_str(model):
    return f'\n==========\nModel Parameters\n\
ALL> {compute_num_params_all(model, text=True)} \n\
TRAINABLE> {compute_num_params(model, text=True)} \n\
NON-TRAINABLE> {compute_num_params(model, text=True, trainable=False)}\n\
                \n==========\n'


def compute_num_params(model, text=True, trainable=True):
    """
    Computes the number of parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        text (bool, optional): If True, returns a formatted string. If False, returns the integer count. Default: True.
        trainable (bool, optional): If True, counts only trainable parameters (requires_grad=True).
                                    If False, counts only non-trainable parameters (requires_grad=False).
                                    If None, counts all parameters regardless of their trainability. Default: True.

    Returns:
        str or int: Formatted string with parameter count or the integer count itself.
    """
    if trainable:
        # Count only trainable parameters
        parameters = [p for p in model.parameters() if p.requires_grad]
    elif not trainable:
        # Count only non-trainable parameters
        parameters = [p for p in model.parameters() if not p.requires_grad]
    else:
        # Count all parameters
        parameters = list(model.parameters())
    
    tot = int(sum([np.prod(p.shape) for p in parameters]))
    
    if text:
        if tot >= 1e6:
            s = '{:.1f}M'.format(tot / 1e6)
        elif tot >= 1e3:
            s = '{:.1f}K'.format(tot / 1e3)
        else:
            s = f'{tot}'
        return f'{s} ({tot})'
    else:
        return tot


def make_optimizer(params, optimizer_spec, load_sd=False):
    optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW,
    }[optimizer_spec['name']](params, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_lr_scheduler(optimizer, lr_scheduler_spec):
    lr_scheduler = {
        'multistep': MultiStepLR,
        'cosine': CosineAnnealingLR,
    }[lr_scheduler_spec['name']](optimizer, **lr_scheduler_spec['args'])
    return lr_scheduler


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class EpochTimer():

    def __init__(self, max_epoch):
        self.max_epoch = max_epoch
        self.epoch = 0
        self.t_start = time.time()
        self.t_last = self.t_start

    def epoch_done(self):
        t_cur = time.time()
        self.epoch += 1
        epoch_time = t_cur - self.t_last
        tot_time = t_cur - self.t_start
        est_time = tot_time / self.epoch * self.max_epoch
        self.t_last = t_cur
        return time_text(epoch_time), time_text(tot_time), time_text(est_time)


class MutlipleImageHandler():
    def __init__(self, dataloader):
        self.n_imgs = len(dataloader)
        self.lst = [0.] * self.n_imgs

    def len(self):
        return self.n_imgs

    def add(self, idx, target):
        self.lst[idx] = target

    def load(self, idx):
        return self.lst[idx]


def time_text(sec):
    if sec >= 3600:
        return f'{sec / 3600:.1f}h'
    elif sec >= 60:
        return f'{sec / 60:.1f}m'
    else:
        return f'{sec:.1f}s'


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)

def input_with_timeout(prompt, timeout):
    result = [None]
    
    def get_input():
        result[0] = input(prompt)
    
    input_thread = threading.Thread(target=get_input)
    input_thread.daemon = True
    input_thread.start()
    input_thread.join(timeout)
    
    if input_thread.is_alive():
        print("\nNo input received. Stopping training automatically.")
        result[0] = 'y'  # Automatically set to 'y' to stop the training
    
    return result[0]
