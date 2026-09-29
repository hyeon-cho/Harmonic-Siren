import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from einops import rearrange
from torchvision.utils import save_image
from tqdm import tqdm

import utils
from .base_trainer_FINAL import BaseTrainer
from .trainers import register


@register('float_trainer_final')
class FloatTrainer(BaseTrainer):
    """
    Float-domain trainer:
    - expects rgb target as continuous channels (no bit-plane targets)
    - disables hard thresholding and BER-dependent stop criteria
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.threshold_fn = utils.thresholding.Identity()
        train_wrapper_args = cfg.datasets.train.wrapper.args
        self.float_bipolar_out = bool(train_wrapper_args.get('bipolar_out', True))
        if not self.float_bipolar_out:
            self.outer_most_non_linearity = lambda x: x

    def _query_to_image(self, q_tensor, height=0, add_factor=1, div_factor=2):
        q_tensor = rearrange(q_tensor, 'batch (h w) rgb -> batch rgb h w', h=height)
        if self.float_bipolar_out:
            q_tensor = (q_tensor + 1.0) / 2.0
        return q_tensor.clamp(0.0, 1.0)

    def valid_step(self, model, batch, patch_size=[128, 128]):
        model.eval()
        valid_psnr = utils.utils.Averager()
        valid_ssim = utils.utils.Averager()
        valid_lpips = utils.utils.Averager()

        full_output = None
        full_gt = None
        last_coord = None
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
            psnr = utils.metrics.calc_psnr(full_output, full_gt)
            ssim = utils.metrics.calc_ssim(full_output, full_gt)
            lpips_val = self._compute_lpips(full_output[0], full_gt[0])

            valid_psnr.add(psnr)
            valid_ssim.add(ssim)
            valid_lpips.add(lpips_val)
            last_coord = full_data['coord']
            self.log_image('valid_output/org', full_output[0].numpy())

            if self.enable_wandb and not self.gt_logged:
                wandb.log(
                    {'images/ground_truth': wandb.Image(np.clip(self._to_hwc_numpy(full_gt[0]), 0, 1))},
                    step=self.epoch,
                )
                self.gt_logged = True

        return valid_psnr, valid_ssim, valid_lpips, full_output[0], full_gt[0], last_coord

    def run_train(self, model):
        cfg = self.cfg
        max_iteration = cfg['max_iteration']
        iteration_per_epoch = len(self.loaders['train'])
        max_epoch = max_iteration // iteration_per_epoch
        eval_epoch = cfg['eval_epoch']

        best_metric = float('-inf')
        self.epoch = iteration_per_epoch
        start_epoch = self.epoch
        epoch_timer = utils.utils.EpochTimer(max_epoch - start_epoch + 1)
        patch_size = self.cfg.datasets.train.wrapper.args.sr_config.patch_size

        for epoch in range(start_epoch, max_epoch + 1, iteration_per_epoch):
            self.log_buffer = [f'Epoch {epoch}']

            train_loss = self.train_step(model, self.loaders['train'])
            self.log_buffer.append(f'train: loss={train_loss.item():.4f}')
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if epoch % eval_epoch == 0:
                valid_psnr, valid_ssim, valid_lpips, img, gt_img, valid_coord = self.valid_step(
                    model, self.loaders['valid'], patch_size
                )
                train_psnr = self.train_eval_step(model, patch_size)

                self.metric_history['epochs'].append(epoch)
                self.metric_history['full_psnr'].append(valid_psnr.item())
                self.metric_history['ssim'].append(valid_ssim.item())
                self.metric_history['lpips'].append(valid_lpips.item())
                self.metric_history['train_psnr'].append(train_psnr.item())
                self.metric_history['test_psnr'].append(0.0)

                self.log_buffer.append(
                    'valid: '
                    f'psnr={valid_psnr.item():.4f}, '
                    f'train_psnr={train_psnr.item():.4f}, '
                    f'ssim={valid_ssim.item():.4f}, '
                    f'lpips={valid_lpips.item():.4f}'
                )

                if valid_psnr.item() > best_metric:
                    best_metric = valid_psnr.item()
                    self.save_ckpt(model, 'float_best.pth')
                    save_image(img, os.path.join(self.cfg._env.save_dir, 'float_best.png'))

                if self.enable_wandb:
                    log_dict = {
                        'epoch': epoch,
                        'loss': train_loss.item(),
                        'full_psnr': valid_psnr.item(),
                        'ssim': valid_ssim.item(),
                        'lpips': valid_lpips.item(),
                        'train_psnr': train_psnr.item(),
                        'test_psnr': 0.0,
                    }
                    if len(self.optimizer.param_groups) > 1:
                        log_dict['learning_rate_muon'] = self.optimizer.param_groups[0]['lr']
                        log_dict['learning_rate_aux'] = self.optimizer.param_groups[1]['lr']
                    else:
                        log_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']

                    if self.log_image_evolution:
                        pred_hwc = self._to_hwc_numpy(img)
                        gt_hwc = self._to_hwc_numpy(gt_img)
                        comparison_fig = self._create_comparison_image(
                            gt_hwc,
                            pred_hwc,
                            epoch,
                            valid_psnr.item(),
                            valid_ssim.item(),
                            valid_lpips.item(),
                        )
                        log_dict.update(
                            {
                                'images/comparison': wandb.Image(
                                    comparison_fig,
                                    caption=(
                                        f'Epoch {epoch}: '
                                        f'PSNR={valid_psnr.item():.2f}, '
                                        f'SSIM={valid_ssim.item():.4f}, '
                                        f'LPIPS={valid_lpips.item():.4f}'
                                    ),
                                ),
                                'images/reconstruction': wandb.Image(
                                    np.clip(pred_hwc, 0, 1),
                                    caption=f'Reconstruction at epoch {epoch}',
                                ),
                            }
                        )
                        plt.close(comparison_fig)

                    train_coords = self.last_train_coords
                    if train_coords is None:
                        train_coords = valid_coord.to(self.device)
                    self._update_layer_metrics(model=model, log_dict=log_dict, train_coords=train_coords)
                    self._update_frequency_metrics(
                        model=model,
                        epoch=epoch,
                        valid_coord=valid_coord,
                        pred_img=img,
                        gt_img=gt_img,
                        log_dict=log_dict,
                    )
                    wandb.log(log_dict, step=self.epoch)
                elif self.log_frequency_dynamics:
                    self._update_frequency_metrics(
                        model=model,
                        epoch=epoch,
                        valid_coord=valid_coord,
                        pred_img=img,
                        gt_img=gt_img,
                        log_dict=None,
                    )

            epoch_time, tot_time, est_time = epoch_timer.epoch_done()
            self.log_buffer.append(f'{epoch_time} {tot_time}/{est_time}')
            self.log(', '.join(self.log_buffer))
            self.epoch += iteration_per_epoch
