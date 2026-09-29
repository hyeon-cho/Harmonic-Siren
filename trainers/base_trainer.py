import os
import copy
import random
import datetime

import yaml
import utils.loss_fn
import wandb
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader

import dataconf
import models
import torch
import utils
from .trainers import register
from einops import rearrange
from torchvision.utils import save_image

from models.activations import BoundedMish, AperiodicSine


@register('base_trainer_final')
class BaseTrainer():

    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        env = cfg._env

        logger, writer = utils.set_save_dir(env.save_dir, replace=False)
        with open(os.path.join(env.save_dir, 'cfg.yaml'), 'w') as f:
            yaml.dump(self.cfg_dict, f, sort_keys=False)
        self.log = logger.info
        self.enable_tb = True
        self.writer = writer

        if env.wandb:
            self.enable_wandb = True
            os.environ['WANDB_NAME'] = env.exp_name
            os.environ['WANDB_DIR'] = env.save_dir
            wandb.init(project=os.environ.get('WANDB_PROJECT', 'ReSIN'),
                       entity=os.environ.get('WANDB_ENTITY'), config=self.cfg_dict)
            now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            wandb.run.name = f'{env.exp_name}_{now}'
            wandb.run.save()
        else:
            self.enable_wandb = False

        # Recomposition at inference must use the same bit-coding as the data
        # wrapper (cfg: datasets.train.wrapper.args.transformer, binary|gray);
        # a fixed GrayTransform here would silently mis-decode NBC-trained runs.
        transformer_key = OmegaConf.select(
            cfg, 'datasets.train.wrapper.args.transformer', default='binary')
        transform_classes = {
            'binary': utils.discretization.BiTransform,
            'gray':   utils.discretization.GrayTransform,
        }
        self.transformation_fn        = transform_classes[transformer_key]()
        self.threshold_fn             = utils.thresholding  .Polarize()

        outer_most_non_linearities = {  # outer most non-linearity
            'linear': lambda x: x,
            'tanh':   lambda x: torch.tanh(x) / 2,
            'asin':   lambda x: AperiodicSine()(x),
        }
        loss_wrappers = {  # Wrapper for loss
            'identity': lambda t: t,
            'exp':      lambda t:  torch.exp(t),
            'pow':      lambda t:  torch.pow(t, 3) ,
        }
        outer_key   = cfg.get('outer_non_linearity', 'asin')
        wrapper_key = cfg.get('loss_wrapper', 'pow')
        self.outer_most_non_linearity = outer_most_non_linearities[outer_key]
        self.loss_wrapper             = loss_wrappers[wrapper_key]

        self.mode = cfg.get('trainer_mode', 'resin')
        if self.mode == 'resin':
            self.train_loss_fn = self.get_loss
        elif self.mode == 'resin_w':
            self.train_loss_fn = self.get_bypassed_loss
            self._modify = list(cfg.get('resin_w_modify', [0, 1, 2]))
            self._scale  = list(cfg.get('resin_w_scale',  [0, 1/2**7, 1/2**6]))
            assert len(self._modify) == len(self._scale), \
                f"resin_w_modify and resin_w_scale length mismatch: {len(self._modify)} vs {len(self._scale)}"
        elif self.mode == 'rec':
            self.train_loss_fn = self.get_rec_loss
        elif self.mode == 'mse':
            self.train_loss_fn = self.get_mse_loss
        else:
            raise ValueError(f"Unknown trainer_mode={self.mode}")

        self.device = torch.device('cuda')
        resinw_str = (f' resin_w_modify={self._modify} resin_w_scale={self._scale}'
                    if self.mode == 'resin_w' else '')
        self.log(f'Environment setup done. trainer_mode={self.mode} outer={outer_key} wrapper={wrapper_key}{resinw_str}')

    ##############################################
    # Basic Utilities 
    ##############################################

    def seed_everything(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed) 

    def register_loss_fn(self, loss_cfg):
        if loss_cfg == 'ReSINLoss':
            loss_fn = utils.loss_fn.EnergyCosineSimilarity()
        elif loss_cfg == 'MSE':
            loss_fn = utils.loss_fn.MSEloss()
        elif loss_cfg == 'BCE':
            loss_fn = utils.loss_fn.BinaryCrossEntropyLoss()
        elif loss_cfg == 'CE':
            loss_fn = utils.loss_fn.CrossEntropyLoss()
        elif loss_cfg == 'InnerProd':
            loss_fn = utils.loss_fn.InnerProductLoss()
        else:
            raise NotImplementedError

        self.loss_fn = loss_fn

    def make_dataset(self, dataset_cfg, tag='train'):
        dataset     = dataconf.make(dataset_cfg['wrapper'])
        self.log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            self.log('  {}: shape={}'.format(k, tuple(v.shape)))

        return dataset

    def make_dataloader(self, dataset, loader_cfg):
        loader_cfg = loader_cfg['loader']
        return DataLoader(dataset, 
                          batch_size=loader_cfg['batch_size'], 
                          shuffle=loader_cfg['shuffle'], 
                          num_workers=loader_cfg['num_workers'],
                          pin_memory=True,)

    def prepare_dataloader(self, dataset_cfg):
        self.log(f'\n\n========= TRAIN DATASET =========')
        train_dataset = self.make_dataset(dataset_cfg['train'], 'train')
        train_dataloader = self.make_dataloader(train_dataset, dataset_cfg['train'])
        self.log(f'\n\n========= VALID DATASET =========')
        valid_dataset = self.make_dataset(dataset_cfg['valid'], 'valid')
        valid_dataloader = self.make_dataloader(valid_dataset, dataset_cfg['valid'])
        self.loaders = {'train': train_dataloader, 'valid': valid_dataloader}

    def make_model(self, model_cfg):
        model = models.make(model_cfg).to(self.device)
        return model

    def make_optimizer(self, model, optimizer_cfg, lr_scheduler_cfg):
        self.optimizer = utils.utils.make_optimizer(model.parameters(), optimizer_cfg) # contains lr, (betas and eps) for optinal
        if lr_scheduler_cfg['name'] is not None:
            self.lr_scheduler = utils.utils.make_lr_scheduler(self.optimizer, lr_scheduler_cfg)
        else:
            self.lr_scheduler = None
        self.log(f'>> Optimizer: {optimizer_cfg["name"]} <<')
        if self.lr_scheduler is not None:
            self.log(f'>> LR Scheduler: {lr_scheduler_cfg["name"]} <<')
        else: 
            self.log(f'>> LR Scheduler: None <<')

    ##############################################
    # Model Feed forward 
    ##############################################

    def get_loss(self, model, coordinate, ground_truth):
        hidden_feature = model.get_hidden_features(model.get_in_features(coordinate))
        prediction     = model.get_out_features   (hidden_feature)
        prediction = self.outer_most_non_linearity(prediction)

        loss_1 = self.loss_fn.calculate_energy(
            rearrange(ground_truth, 'batch query channel -> batch (query channel)'),
            rearrange(prediction,   'batch query channel -> batch (query channel)'))

        loss = self.loss_wrapper(loss_1) 
        return loss , loss.item() 

    def get_bypassed_loss(self, model, coordinate, ground_truth):
        prediction = model(coordinate)
        prediction = self.outer_most_non_linearity(prediction)

        _pred = rearrange(prediction,   'batch query (rgb bit) -> batch bit query rgb', rgb=3)
        _gt   = rearrange(ground_truth, 'batch query (rgb bit) -> batch bit query rgb', rgb=3)
        loss = 0.
        for bit_idx in range(8):
            _loss = self.loss_fn.calculate_energy(
                rearrange(_pred[:, bit_idx, :, :], 'batch query rgb -> batch (query rgb)'),
                rearrange(_gt[:, bit_idx, :, :], 'batch query rgb -> batch (query rgb)')
            )
            if bit_idx in self._modify:
                _loss = _loss * self._scale[bit_idx]
            loss = loss + _loss

        loss = loss / (8 - len(self._modify) + sum(self._scale))
        loss = self.loss_wrapper(loss)
        return loss, loss.item()

    def get_rec_loss(self, model, coordinate, ground_truth):
        if hasattr(model, 'get_in_features'):
            hidden_feature = model.get_hidden_features(model.get_in_features(coordinate))
            prediction     = model.get_out_features   (hidden_feature)
        else:
            prediction = model(coordinate)
        prediction = self.outer_most_non_linearity(prediction)

        loss_1 = self.loss_fn.calculate_energy(
            rearrange(ground_truth, 'batch query channel -> batch (query channel)'),
            rearrange(prediction,   'batch query channel -> batch (query channel)'))

        loss = self.loss_wrapper(loss_1)
        return loss, loss.item()

    def get_mse_loss(self, model, coordinate, ground_truth):
        # Conventional INR pipeline: linear outer + MSE on raw RGB targets in [0,1].
        prediction = model(coordinate)
        prediction = self.outer_most_non_linearity(prediction)
        loss = self.loss_fn.calculate_energy(
            rearrange(ground_truth, 'batch query channel -> batch (query channel)'),
            rearrange(prediction,   'batch query channel -> batch (query channel)'))
        loss = self.loss_wrapper(loss)
        return loss, loss.item()

    def _query_to_image(self, q_tensor, height=0,
                       add_factor=1, div_factor=2):
        if self.mode == 'rec':
            q_tensor = rearrange(q_tensor, 'batch (h w) c -> batch c h w', h=height)
            return ((q_tensor + add_factor) / div_factor).clamp(0, 1)
        if self.mode == 'mse':
            # Target/prediction already in [0,1]; just rearrange.
            return rearrange(q_tensor, 'batch (h w) c -> batch c h w', h=height)
        q_tensor = rearrange(q_tensor,
                             'batch (h w) (rgb bit) -> batch bit rgb h w',
                             rgb=3, h=height)
        q_tensor = self.transformation_fn.composition((q_tensor + add_factor) / div_factor)
        return q_tensor

    def get_images(self, model, coordinate, ground_truth, shape=[256,256]):
        pred = model(coordinate)
        pred = self.outer_most_non_linearity(pred)
        if self.mode == 'mse':
            pred = pred.clamp(0, 1)
        elif self.mode != 'rec':
            pred = self.threshold_fn.thresholding(pred)

        prediction   = self._query_to_image(pred, height=shape[0])
        ground_truth = self._query_to_image(ground_truth, height=shape[0])

        return prediction, ground_truth

    ##############################################
    # Model Steps
    ##############################################

    def train_step(self, model, train_loader):
        model.train()
        total_loss = utils.utils.Averager()
        pbar = tqdm(train_loader, leave=False)

        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.to(self.device)

            self.optimizer.zero_grad()
            loss, loss_log = self.train_loss_fn(model, batch['coord'], batch['rgb']); total_loss.add(loss.item())
            loss.backward()
            self.optimizer.step()
            pbar.set_description(f'train: loss={loss_log:.4f}')
        return total_loss

    def valid_step(self, model, batch, patch_size=[128,128]):
        model.eval()
        valid_psnr  = utils.utils.Averager()
        valid_ber   = utils.utils.Averager()
        valid_ssim  = utils.utils.Averager()
        valid_lpips = utils.utils.Averager()

        n_inf = 0
        full_output = None
        pbar = tqdm(self.loaders['valid'], leave=False, desc='valid')
        for batch in pbar:
            full_data = {
                'coord': batch['coord'],
                'rgb': batch['rgb'],
                'height': int(batch['inp_size'][0][0].item()),
                'width': int(batch['inp_size'][0][1].item()),
            }
            with torch.no_grad():
                full_output = self.patchwise_feedforward(model, full_data, patch_size)

            full_gt = self._query_to_image(full_data['rgb'], height=full_data['height'])

            psnr         = utils.metrics.calc_psnr(full_output, full_gt)
            ber, ber_lst = utils.metrics.calculate_ber(full_output, full_gt)
            ssim         = utils.metrics.calc_ssim(full_output, full_gt)
            lpips_val    = utils.metrics.calc_lpips(full_output, full_gt)

            if self.mode not in ('rec', 'mse') and ber == 0:
                n_inf = n_inf + 1
                psnr = 0          # set psnr 0 for stop indicator

            valid_ber.add(ber)
            valid_psnr.add(psnr)
            valid_ssim.add(ssim)
            valid_lpips.add(lpips_val)
            self.log_image('valid_output/org', full_output[0].numpy())

        return valid_psnr, valid_ber, valid_ssim, valid_lpips, ber_lst, n_inf, full_output[0]

    def patchwise_feedforward(self, model, full_data, patch_size=[128,128]):
        full_coord  = full_data['coord']
        full_rgb    = full_data['rgb']
        full_output = torch.zeros_like(full_rgb)[:,:, :3]
        height      = full_data['height']
        width       = full_data['width']

        with torch.no_grad():
            _patch_iteration= int((height * width) / (patch_size[0] * patch_size[1]))
            for patch in range(_patch_iteration): 
                _from = (patch_size[0]*patch_size[1])*patch
                _to   = (patch_size[0]*patch_size[1])*(patch+1)
                patch_coord, patch_rgb = full_coord[:, _from:_to:], full_rgb[:, _from:_to:]
                patch_prediction, patch_rgb = self.get_images(model, 
                                                                patch_coord.cuda(), 
                                                                patch_rgb.cuda(), 
                                                                shape=patch_size)

                patch_prediction = patch_prediction.cpu().detach(); patch_rgb = None;
                full_output[:, _from:_to:] = rearrange(patch_prediction, 'batch channel height width -> batch (height width) channel')
                patch_prediction = None 

            full_output = (rearrange(full_output, 'batch (h w) rgb -> batch rgb h w', h=height)).cpu().detach()
        return full_output

    ##############################################
    # Main Loop 
    ##############################################

    def run_train(self, model): 
        cfg = self.cfg 
        max_iteration       = cfg['max_iteration']
        iteration_per_epoch = len(self.loaders['train'])
        max_epoch           = max_iteration // iteration_per_epoch 

        eval_epoch          = cfg['eval_epoch']

        best_metric  = float('-inf')
        self.epoch   = iteration_per_epoch
        start_epoch  = self.epoch
        epoch_timer  = utils.utils.EpochTimer(max_epoch - start_epoch + 1)
        for epoch in range(start_epoch, max_epoch + 1, iteration_per_epoch):
            self.log_buffer = [f'Epoch {epoch}']

            # [1] Train
            train_loss = self.train_step(model, self.loaders['train'])
            self.log_buffer.append(f'train: loss={train_loss.item():.4f}')
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # [2] Evaluation
            if epoch % eval_epoch == 0:
                valid_psnr, valid_ber, valid_ssim, valid_lpips, ber_lst, n_inf, img = self.valid_step(model, self.loaders['valid'], self.cfg.datasets.train.wrapper.args.sr_config.patch_size)
                self.log_buffer.append(
                    f'valid: psnr={valid_psnr.item():.4f}, ssim={valid_ssim.item():.4f}, '
                    f'lpips={valid_lpips.item():.4f}, ber={valid_ber.item():.4f}, n_inf={n_inf}'
                )
                if valid_psnr.item() > best_metric:
                    best_metric = valid_psnr.item()
                    self.save_ckpt(model, 'finite_best.pth')
                    save_image(img, os.path.join(self.cfg._env.save_dir, 'finite_best.png'))
                if self.mode not in ('rec', 'mse') and valid_psnr.item() == 0:
                    self.save_ckpt(model, 'infinite.pth')
                    save_image(img, os.path.join(self.cfg._env.save_dir, 'infinite.png'))

            epoch_time, tot_time, est_time = epoch_timer.epoch_done()
            self.log_buffer.append(f'{epoch_time} {tot_time}/{est_time}')
            self.log(', '.join(self.log_buffer))
            if self.mode not in ('rec', 'mse') and valid_psnr.item() == 0: self.training_stop_callback()
            self.epoch += (iteration_per_epoch)

    ##############################################
    # Misc
    ##############################################

    def save_ckpt(self, model, filename):
        model_spec = copy.copy(self.cfg_dict['model'])
        model_spec['sd'] = model.state_dict()
        spec = copy.copy(self.cfg_dict['optimizer'])
        spec['sd'] = self.optimizer.state_dict()
        optimizer_spec = spec
        ckpt = {
            'model': model_spec,
            'optimizers': optimizer_spec,
            'epoch': self.epoch,
        }; torch.save(ckpt, os.path.join(self.cfg._env.save_dir, filename))

    def log_image(self, k, v):
        if self.enable_tb:
            self.writer.add_image(k, v, global_step=self.epoch)
        if self.enable_wandb:
            wandb.log({k: wandb.Image(v.transpose(1,2,0))}, step=self.epoch)

    def log_scaler(self, k, v):
        if self.enable_tb:
            self.writer.add_scalar(k, v, global_step=self.epoch)
        if self.enable_wandb:
            wandb.log({k: v}, step=self.epoch)

    def training_stop_callback(self, default_value='y', timeout=2):
        stop = utils.utils.input_with_timeout('Training is stopped due to the infinite PSNR. Terminate? [y/n]', timeout).lower() == default_value
        if stop: 
            self.loaders=None
            raise SystemExit('Training is stopped due to the infinite PSNR.')
        else:
            pass

    def run(self): 
        if self.cfg.random_seed is not None:
            self.seed_everything(self.cfg.random_seed)

        self.prepare_dataloader(self.cfg.datasets)
        model = self.make_model(self.cfg.model)
        print(utils.get_model_param_str(model))
        self.make_optimizer(model, self.cfg.optimizer, self.cfg.optimizer.schedulers)
        self.register_loss_fn(self.cfg.loss)
        self.run_train(model)
