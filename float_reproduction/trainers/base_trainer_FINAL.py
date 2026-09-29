import os
import copy
import random
import datetime

import matplotlib.pyplot as plt
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
from models.frequency_logging import (
    compute_line_spectrum_concentration,
    compute_residual_frequency_alignment,
    compute_target_frequency_alignment,
    extract_input_frequency_state,
    save_frequency_snapshot,
    summarize_frequency_state,
)
from models.recurrent_parameter_metrics import recurrent_parameter_dynamics_info

try:
    import lpips
except ImportError:
    lpips = None


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
            wandb_project = os.environ.get('WANDB_PROJECT', None)
            wandb_entity = os.environ.get('WANDB_ENTITY', None)
            try:
                wandb.login(relogin=False)
            except Exception as exc:
                self.log(f'WARNING: wandb.login failed ({exc}). Trying wandb.init with existing local settings.')
            wandb.init(project=wandb_project, entity=wandb_entity, config=self.cfg_dict)
            now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            wandb.run.name = f'{env.exp_name}_{now}'
            wandb.run.save()
        else:
            self.enable_wandb = False

        self.transformation_fn        = utils.discretization.GrayTransform()
        self.threshold_fn             = utils.thresholding  .Polarize() 

        outer_most_non_linearities = {  # outer most non-linearity
            'linear': lambda x: x,
            'tanh':   lambda x: torch.tanh(x) / 2,
            'asin':   lambda x: AperiodicSine()(x),
        }; self.outer_most_non_linearity = outer_most_non_linearities['asin']
        loss_wrappers = {  # Wrapper for loss
            'identity': lambda t: t,
            'exp':      lambda t:  torch.exp(t),
            'pow':      lambda t:  torch.pow(t, 3) ,
        }; self.loss_wrapper = loss_wrappers['identity']

        # For GAM 
        self.train_loss_fn = self.get_loss

        self.device = torch.device('cuda')
        self.log_image_evolution = bool(cfg.get('log_image_evolution', True))
        self.log_jacobian_metrics = bool(cfg.get('log_jacobian_metrics', True))
        self.log_recurrent_param_metrics = bool(cfg.get('log_recurrent_param_metrics', True))
        self.jacobian_max_samples = max(1, int(cfg.get('jacobian_max_samples', 1024)))
        self.jacobian_layer_max_samples = max(1, int(cfg.get('jacobian_layer_max_samples', 32)))
        self.log_frequency_dynamics = bool(cfg.get('log_frequency_dynamics', False))
        self.frequency_log_interval = max(1, int(cfg.get('frequency_log_interval', 1)))
        self.frequency_traj_channels = max(1, int(cfg.get('frequency_traj_channels', 8)))
        self.frequency_fft_channels = max(1, int(cfg.get('frequency_fft_channels', 16)))
        self.frequency_radial_bins = max(8, int(cfg.get('frequency_radial_bins', 64)))
        self.frequency_save_snapshots = bool(cfg.get('frequency_save_snapshots', True))
        self.frequency_hit_delta = float(cfg.get('frequency_hit_delta', 0.25))
        target_freq_cfg = cfg.get('frequency_target_frequencies', None)
        if target_freq_cfg is None:
            self.frequency_target_frequencies = np.array([], dtype=np.float64)
        else:
            self.frequency_target_frequencies = np.asarray(target_freq_cfg, dtype=np.float64).reshape(-1)
        self.frequency_log_dir = os.path.join(env.save_dir, 'frequency_logs')
        self._frequency_traj_channel_ids = None
        self._frequency_initial_cycles = None
        self._frequency_initial_omega = None
        self._residual_js_initial = None
        self._residual_js_best = None
        if self.log_frequency_dynamics and self.frequency_save_snapshots:
            os.makedirs(self.frequency_log_dir, exist_ok=True)
        self.layer_metrics = {}
        self.metric_history = {
            'epochs': [],
            'full_psnr': [],
            'ssim': [],
            'lpips': [],
            'train_psnr': [],
            'test_psnr': [],
        }
        self.last_train_coords = None
        self.gt_logged = False
        self.lpips_model = self._build_lpips_model()
        self.log(f'Environment setup done.')

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

    def _build_lpips_model(self):
        if lpips is None:
            self.log('WARNING: lpips package not found. LPIPS logging is disabled.')
            return None
        try:
            return lpips.LPIPS(net='alex').to(self.device)
        except Exception as exc:
            self.log(f'WARNING: LPIPS model initialization failed ({exc}). LPIPS logging is disabled.')
            return None

    def _to_hwc_numpy(self, image_chw):
        return image_chw.detach().cpu().permute(1, 2, 0).numpy()

    def _compute_lpips(self, pred_img, gt_img):
        if self.lpips_model is None:
            return float('nan')
        pred = pred_img.unsqueeze(0).to(self.device).float().clamp(0.0, 1.0)
        gt = gt_img.unsqueeze(0).to(self.device).float().clamp(0.0, 1.0)
        pred = pred * 2.0 - 1.0
        gt = gt * 2.0 - 1.0
        with torch.no_grad():
            return float(self.lpips_model(pred, gt).item())

    def _create_comparison_image(self, gt_img, recon_img, epoch, psnr_val, ssim_val, lpips_val):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(gt_img)
        axes[0].set_title('Ground Truth', fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(np.clip(recon_img, 0, 1))
        axes[1].set_title(f'Reconstruction\nEpoch {epoch}', fontsize=12)
        axes[1].axis('off')

        diff = np.abs(gt_img - recon_img)
        axes[2].imshow(np.clip(diff, 0, 1), cmap='viridis')
        axes[2].set_title(
            f'Difference \nPSNR: {psnr_val:.2f}dB\nSSIM: {ssim_val:.4f}\nLPIPS: {lpips_val:.4f}',
            fontsize=12,
        )
        axes[2].axis('off')
        plt.tight_layout()
        return fig

    def _append_metric(self, key, value):
        if key not in self.layer_metrics:
            self.layer_metrics[key] = []
        self.layer_metrics[key].append(value)

    def _init_frequency_tracking(self, model):
        if not self.log_frequency_dynamics:
            return
        freq_state = extract_input_frequency_state(model)
        if freq_state is None:
            return

        summary = summarize_frequency_state(freq_state)
        omega_cycles = summary['omega_cycles']
        if omega_cycles.size == 0:
            return

        n_traj = min(self.frequency_traj_channels, omega_cycles.size)
        self._frequency_traj_channel_ids = np.linspace(0, omega_cycles.size - 1, num=n_traj, dtype=np.int64)
        self._frequency_initial_cycles = omega_cycles[self._frequency_traj_channel_ids].copy()
        self._frequency_initial_omega = freq_state['omega'].copy()
        self._residual_js_initial = None
        self._residual_js_best = None

        if self.frequency_save_snapshots:
            try:
                save_frequency_snapshot(
                    out_dir=self.frequency_log_dir,
                    epoch=0,
                    freq_state=freq_state,
                    extra={
                        'omega_cycles': omega_cycles.astype(np.float32),
                        'omega_norm': summary['omega_norm'].astype(np.float32),
                        'weight_norm': summary['weight_norm'].astype(np.float32),
                        'traj_channel_ids': self._frequency_traj_channel_ids.astype(np.int64),
                        'traj_initial_cycles': self._frequency_initial_cycles.astype(np.float32),
                    },
                )
            except Exception as exc:
                self.log(f'WARNING: Initial frequency snapshot save failed: {exc}')

    def _update_frequency_metrics(self, model, epoch, valid_coord, pred_img, gt_img, log_dict=None):
        if not self.log_frequency_dynamics:
            return
        if epoch % self.frequency_log_interval != 0:
            return

        freq_state = extract_input_frequency_state(model)
        if freq_state is None:
            return
        summary = summarize_frequency_state(freq_state)
        omega_cycles = summary['omega_cycles']

        if self._frequency_traj_channel_ids is None or self._frequency_initial_cycles is None:
            self._init_frequency_tracking(model)

        def _log_scalar(key, value):
            if log_dict is not None:
                log_dict[key] = float(value)

        _log_scalar('omega/input_scale', summary['omega_in'])
        _log_scalar('omega/n_channels', summary['num_channels'])
        _log_scalar('omega/input_dim', summary['input_dim'])
        _log_scalar('omega/weight_norm_mean', summary['weight_norm_mean'])
        _log_scalar('omega/weight_norm_median', summary['weight_norm_median'])
        _log_scalar('omega/weight_norm_p90', summary['weight_norm_p90'])
        _log_scalar('omega/norm_mean', summary['omega_norm_mean'])
        _log_scalar('omega/norm_median', summary['omega_norm_median'])
        _log_scalar('omega/norm_p90', summary['omega_norm_p90'])
        _log_scalar('omega/norm_max', summary['omega_norm_max'])
        _log_scalar('omega/cycles_mean', summary['omega_cycles_mean'])
        _log_scalar('omega/cycles_median', summary['omega_cycles_median'])
        _log_scalar('omega/cycles_p90', summary['omega_cycles_p90'])
        _log_scalar('omega/cycles_max', summary['omega_cycles_max'])
        _log_scalar(
            'omega_comp/cycles_median_over_input_scale',
            summary['omega_cycles_median'] / (abs(summary['omega_in']) + 1e-12),
        )
        _log_scalar(
            'omega_comp/weight_norm_median_times_input_scale',
            summary['weight_norm_median'] * abs(summary['omega_in']),
        )
        _log_scalar(
            'omega_comp/weight_norm_mean_times_input_scale',
            summary['weight_norm_mean'] * abs(summary['omega_in']),
        )

        if 'omega_scalar_cycles_mean' in summary:
            _log_scalar('omega/scalar_cycles_mean', summary['omega_scalar_cycles_mean'])
            _log_scalar('omega/scalar_cycles_median', summary['omega_scalar_cycles_median'])

        direction_cos = None
        if (
            self._frequency_initial_omega is not None
            and self._frequency_initial_omega.shape == freq_state['omega'].shape
            and freq_state['omega'].ndim == 2
        ):
            omega_now = freq_state['omega'].astype(np.float64)
            omega_init = self._frequency_initial_omega.astype(np.float64)
            dot = np.sum(omega_now * omega_init, axis=1)
            now_norm = np.linalg.norm(omega_now, axis=1)
            init_norm = np.linalg.norm(omega_init, axis=1)
            denom = now_norm * init_norm
            valid = denom > 1e-12
            direction_cos = np.full_like(denom, np.nan, dtype=np.float64)
            direction_cos[valid] = dot[valid] / denom[valid]
            direction_cos = np.clip(direction_cos, -1.0, 1.0)
            valid_cos = direction_cos[valid]
            if valid_cos.size > 0:
                _log_scalar('omega_direction/cos_mean', float(np.mean(valid_cos)))
                _log_scalar('omega_direction/cos_median', float(np.median(valid_cos)))
                _log_scalar('omega_direction/cos_p10', float(np.percentile(valid_cos, 10)))
                _log_scalar('omega_direction/cos_p90', float(np.percentile(valid_cos, 90)))
                _log_scalar('omega_direction/cos_min', float(np.min(valid_cos)))
                _log_scalar('omega_direction/cos_max', float(np.max(valid_cos)))
                _log_scalar('omega_direction/valid_fraction', float(np.mean(valid.astype(np.float64))))

        if (
            self._frequency_traj_channel_ids is not None
            and self._frequency_initial_cycles is not None
            and self._frequency_traj_channel_ids.size > 0
        ):
            current_traj_cycles = omega_cycles[self._frequency_traj_channel_ids]
            traj_delta = np.abs(current_traj_cycles - self._frequency_initial_cycles)
            _log_scalar('omega_traj/delta_abs_mean', float(np.mean(traj_delta)))
            _log_scalar('omega_traj/delta_abs_max', float(np.max(traj_delta)))
            for i, chan_id in enumerate(self._frequency_traj_channel_ids):
                _log_scalar(f'omega_traj/ch_{int(chan_id)}_cycles', float(current_traj_cycles[i]))
                _log_scalar(f'omega_traj/ch_{int(chan_id)}_delta_abs', float(traj_delta[i]))
                if direction_cos is not None and int(chan_id) < direction_cos.size:
                    cos_val = direction_cos[int(chan_id)]
                    if np.isfinite(cos_val):
                        _log_scalar(f'omega_direction/ch_{int(chan_id)}_cos', float(cos_val))

        if log_dict is not None and self.enable_wandb:
            log_dict['omega/hist_cycles'] = wandb.Histogram(omega_cycles.tolist())
            log_dict['omega/hist_weight_norm'] = wandb.Histogram(summary['weight_norm'].tolist())
            if direction_cos is not None:
                valid_cos = direction_cos[np.isfinite(direction_cos)]
                if valid_cos.size > 0:
                    log_dict['omega_direction/hist_cos'] = wandb.Histogram(valid_cos.tolist())

        if self.frequency_target_frequencies.size > 0:
            align_metrics = compute_target_frequency_alignment(
                omega_cycles=omega_cycles,
                target_frequencies=self.frequency_target_frequencies,
                hit_delta=self.frequency_hit_delta,
            )
            for key, value in align_metrics.items():
                _log_scalar(key, value)

        line_info = {}
        if valid_coord is not None:
            try:
                line_info = compute_line_spectrum_concentration(
                    freq_state=freq_state,
                    coord=valid_coord,
                    max_channels=self.frequency_fft_channels,
                )
            except Exception as exc:
                self.log(f'WARNING: Line-spectrum concentration logging failed: {exc}')

        if line_info:
            _log_scalar('line_spectrum/concentration_mean', line_info['line_spectrum/concentration_mean'])
            _log_scalar('line_spectrum/concentration_median', line_info['line_spectrum/concentration_median'])
            _log_scalar('line_spectrum/concentration_min', line_info['line_spectrum/concentration_min'])
            _log_scalar('line_spectrum/concentration_max', line_info['line_spectrum/concentration_max'])
            _log_scalar('line_spectrum/num_channels', line_info['line_spectrum/num_channels'])
            if log_dict is not None and self.enable_wandb:
                log_dict['line_spectrum/concentration_hist'] = wandb.Histogram(
                    line_info['line_spectrum/concentration_ratios'].tolist()
                )

        residual_info = {}
        if valid_coord is not None and pred_img is not None and gt_img is not None:
            try:
                residual_info = compute_residual_frequency_alignment(
                    pred_img=pred_img,
                    gt_img=gt_img,
                    coord=valid_coord,
                    omega_cycles=omega_cycles,
                    radial_bins=self.frequency_radial_bins,
                )
            except Exception as exc:
                self.log(f'WARNING: Residual spectrum logging failed: {exc}')

        if residual_info:
            _log_scalar('residual_spectrum/corr_omega_hist', residual_info['residual_spectrum/corr_omega_hist'])
            _log_scalar(
                'residual_spectrum/js_divergence_omega_hist',
                residual_info['residual_spectrum/js_divergence_omega_hist'],
            )
            _log_scalar('residual_spectrum/radial_power_mean', residual_info['residual_spectrum/radial_power_mean'])
            _log_scalar('residual_spectrum/radial_power_max', residual_info['residual_spectrum/radial_power_max'])
            js_val = float(residual_info['residual_spectrum/js_divergence_omega_hist'])
            if np.isfinite(js_val):
                if self._residual_js_initial is None:
                    self._residual_js_initial = js_val
                if self._residual_js_best is None:
                    self._residual_js_best = js_val
                else:
                    self._residual_js_best = min(self._residual_js_best, js_val)
                _log_scalar('residual_spectrum/js_divergence_omega_hist_init', self._residual_js_initial)
                _log_scalar(
                    'residual_spectrum/js_divergence_omega_hist_delta_from_init',
                    js_val - self._residual_js_initial,
                )
                _log_scalar(
                    'residual_spectrum/js_divergence_omega_hist_improvement',
                    self._residual_js_initial - js_val,
                )
                _log_scalar('residual_spectrum/js_divergence_omega_hist_best', self._residual_js_best)
            if log_dict is not None and self.enable_wandb:
                log_dict['residual_spectrum/radial_hist'] = wandb.Histogram(
                    residual_info['residual_spectrum/radial_hist'].tolist()
                )
                log_dict['residual_spectrum/omega_hist'] = wandb.Histogram(
                    residual_info['residual_spectrum/omega_hist'].tolist()
                )

        if self.frequency_save_snapshots:
            extra = {
                'omega_cycles': omega_cycles.astype(np.float32),
                'omega_norm': summary['omega_norm'].astype(np.float32),
                'weight_norm': summary['weight_norm'].astype(np.float32),
            }
            if self._frequency_traj_channel_ids is not None:
                extra['traj_channel_ids'] = self._frequency_traj_channel_ids.astype(np.int64)
            if self._frequency_initial_cycles is not None:
                extra['traj_initial_cycles'] = self._frequency_initial_cycles.astype(np.float32)
            if line_info:
                extra['line_concentration_ratios'] = line_info['line_spectrum/concentration_ratios'].astype(np.float32)
                extra['line_channel_ids'] = line_info['line_spectrum/channel_ids'].astype(np.int64)
            if residual_info:
                extra['residual_radial_hist'] = residual_info['residual_spectrum/radial_hist'].astype(np.float32)
                extra['omega_radial_hist'] = residual_info['residual_spectrum/omega_hist'].astype(np.float32)
                extra['residual_radial_bin_edges'] = residual_info['residual_spectrum/radial_bin_edges'].astype(np.float32)
            if direction_cos is not None:
                extra['omega_direction_cos'] = direction_cos.astype(np.float32)
            try:
                save_frequency_snapshot(
                    out_dir=self.frequency_log_dir,
                    epoch=epoch,
                    freq_state=freq_state,
                    extra=extra,
                )
            except Exception as exc:
                self.log(f'WARNING: Frequency snapshot save failed at epoch {epoch}: {exc}')

    def _update_jacobian_metrics(self, model, log_dict, train_coords):
        if train_coords is None:
            return
        if not self.log_jacobian_metrics:
            return
        if not hasattr(model, 'get_jacobian_spectral_info'):
            return

        jacobian_coords = train_coords
        if jacobian_coords.ndim > 2:
            jacobian_coords = jacobian_coords.reshape(-1, jacobian_coords.shape[-1])
        if jacobian_coords.shape[0] > self.jacobian_max_samples:
            sample_idx = torch.randperm(jacobian_coords.shape[0], device=jacobian_coords.device)[:self.jacobian_max_samples]
            jacobian_coords = jacobian_coords[sample_idx]

        jacobian_info = {}
        try:
            jacobian_info = model.get_jacobian_spectral_info(jacobian_coords)
        except Exception as exc:
            self.log(f'WARNING: Jacobian spectral metric computation failed: {exc}')

        metric_pairs = [
            ('jacobian_spectral_norm_mean', 'jacobian_spectral_norm/mean'),
            ('jacobian_spectral_norm_max', 'jacobian_spectral_norm/max'),
            ('jacobian_spectral_norm_min', 'jacobian_spectral_norm/min'),
            ('jacobian_spectral_norm_std', 'jacobian_spectral_norm/std'),
            ('bound_over_jacobian_max', 'jacobian_spectral_norm/bound_over_max'),
        ]
        for metric_key, wandb_key in metric_pairs:
            metric_val = jacobian_info.get(metric_key, float('nan'))
            self._append_metric(metric_key, metric_val)
            log_dict[wandb_key] = metric_val

        model_singular_metric_pairs = [
            ('jacobian_sigma1', 'jacobian_singular_values/model/sigma1'),
            ('jacobian_sigma2', 'jacobian_singular_values/model/sigma2'),
            ('jacobian_sigma_min', 'jacobian_singular_values/model/sigma_min'),
        ]
        for metric_key, wandb_key in model_singular_metric_pairs:
            metric_val = jacobian_info.get(metric_key, float('nan'))
            self._append_metric(metric_key, metric_val)
            log_dict[wandb_key] = metric_val

        num_samples_val = jacobian_info.get('jacobian_num_samples', int(jacobian_coords.shape[0]))
        input_dim_val = jacobian_info.get('jacobian_input_dim', int(jacobian_coords.shape[-1]))
        output_dim_val = jacobian_info.get('jacobian_output_dim', float('nan'))
        self._append_metric('jacobian_num_samples', num_samples_val)
        self._append_metric('jacobian_input_dim', input_dim_val)
        self._append_metric('jacobian_output_dim', output_dim_val)
        log_dict['jacobian_spectral_norm/num_samples'] = num_samples_val
        log_dict['jacobian_spectral_norm/input_dim'] = input_dim_val
        log_dict['jacobian_spectral_norm/output_dim'] = output_dim_val

        if not hasattr(model, 'get_layer_local_jacobian_spectral_info'):
            return

        layer_local_info = {}
        try:
            layer_local_info = model.get_layer_local_jacobian_spectral_info(
                jacobian_coords,
                max_samples=self.jacobian_layer_max_samples,
            )
        except Exception as exc:
            self.log(f'WARNING: Layer-local Jacobian metric computation failed: {exc}')

        layer_infos = layer_local_info.get('layer_local_jacobian_infos', [])
        per_layer_max_vals = []
        for layer_info in layer_infos:
            layer_idx = int(layer_info.get('layer_index', len(per_layer_max_vals)))
            local_mean = layer_info.get('jacobian_spectral_norm_mean', float('nan'))
            local_max = layer_info.get('jacobian_spectral_norm_max', float('nan'))
            local_min = layer_info.get('jacobian_spectral_norm_min', float('nan'))
            local_std = layer_info.get('jacobian_spectral_norm_std', float('nan'))

            self._append_metric(f'local_jacobian_mean_layer_{layer_idx}', local_mean)
            self._append_metric(f'local_jacobian_max_layer_{layer_idx}', local_max)
            self._append_metric(f'local_jacobian_min_layer_{layer_idx}', local_min)
            self._append_metric(f'local_jacobian_std_layer_{layer_idx}', local_std)

            log_dict[f'local_jacobian_spectral_norm/layer_{layer_idx}/mean'] = local_mean
            log_dict[f'local_jacobian_spectral_norm/layer_{layer_idx}/max'] = local_max
            log_dict[f'local_jacobian_spectral_norm/layer_{layer_idx}/min'] = local_min
            log_dict[f'local_jacobian_spectral_norm/layer_{layer_idx}/std'] = local_std

            local_singular_metric_pairs = [
                ('jacobian_sigma1', f'local_jacobian_singular_values/layer_{layer_idx}/sigma1'),
                ('jacobian_sigma2', f'local_jacobian_singular_values/layer_{layer_idx}/sigma2'),
                ('jacobian_sigma_min', f'local_jacobian_singular_values/layer_{layer_idx}/sigma_min'),
            ]
            for metric_key, wandb_key in local_singular_metric_pairs:
                metric_val = layer_info.get(metric_key, float('nan'))
                self._append_metric(f'local_{metric_key}_layer_{layer_idx}', metric_val)
                log_dict[wandb_key] = metric_val

            if not np.isnan(local_max):
                per_layer_max_vals.append(local_max)

        if per_layer_max_vals:
            max_over_layers = float(np.max(per_layer_max_vals))
            mean_over_layers = float(np.mean(per_layer_max_vals))
            self._append_metric('local_jacobian_max_over_layers', max_over_layers)
            self._append_metric('local_jacobian_mean_of_layer_max', mean_over_layers)
            log_dict['local_jacobian_spectral_norm/max_over_layers'] = max_over_layers
            log_dict['local_jacobian_spectral_norm/mean_of_layer_max'] = mean_over_layers

    def _update_layer_metrics(self, model, log_dict, train_coords):
        if hasattr(model, 'get_detailed_matrix_info'):
            info = model.get_detailed_matrix_info()
            for i, layer_info in enumerate(info.get('layer_infos', [])):
                stable_rank_val = layer_info.get('stable_rank', 0.0)
                effective_rank_val = layer_info.get('effective_rank', 0.0)
                spectral_norm_val = layer_info.get('linear_spectral_norm', 0.0)
                condition_number_val = layer_info.get('spectral_condition_no', 0.0)

                self._append_metric(f'stable_rank_layer_{i}', stable_rank_val)
                self._append_metric(f'effective_rank_layer_{i}', effective_rank_val)
                self._append_metric(f'spectral_norm_layer_{i}', spectral_norm_val)
                self._append_metric(f'condition_number_layer_{i}', condition_number_val)

                log_dict[f'stable_rank/layer_{i}'] = stable_rank_val
                log_dict[f'effective_rank/layer_{i}'] = effective_rank_val
                log_dict[f'spectral_norm/layer_{i}'] = spectral_norm_val
                log_dict[f'condition_number/layer_{i}'] = condition_number_val

            if 'end_to_end_spectral_bound' in info:
                end_to_end_val = info['end_to_end_spectral_bound']
                self._append_metric('end_to_end_bound', end_to_end_val)
                log_dict['end_to_end_bound'] = end_to_end_val

        if self.log_recurrent_param_metrics:
            recurrent_param_info = {}
            try:
                recurrent_param_info = recurrent_parameter_dynamics_info(model)
            except Exception as exc:
                self.log(f'WARNING: Recurrent parameter metric computation failed: {exc}')

            for metric_key, metric_val in recurrent_param_info.items():
                hist_key = metric_key.replace('/', '_')
                self._append_metric(hist_key, metric_val)
                log_dict[metric_key] = metric_val

        self._update_jacobian_metrics(model=model, log_dict=log_dict, train_coords=train_coords)

    def register_loss_fn(self, loss_cfg):
        if loss_cfg == 'GAMLoss':
            loss_fn = utils.loss_fn.EnergyCosineSimilarity()
        elif loss_cfg == 'BCE':
            loss_fn = utils.loss_fn.BinaryCrossEntropyLoss()
        elif loss_cfg == 'CE':
            loss_fn = utils.loss_fn.CrossEntropyLoss()
        elif loss_cfg == 'InnerProd':
            loss_fn = utils.loss_fn.InnerProductLoss()
        elif loss_cfg == 'MSELoss':
            loss_fn = utils.loss_fn.MSELoss()
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

    def _query_to_image(self, q_tensor, height=0, 
                       add_factor=1, div_factor=2):
        q_tensor = rearrange(q_tensor, 
                             'batch (h w) (rgb bit) -> batch bit rgb h w', 
                             rgb=3, h=height)

        q_tensor = self.transformation_fn.composition((q_tensor + add_factor) / div_factor)
        return q_tensor

    def get_images(self, model, coordinate, ground_truth, shape=[256,256]):
        pred = model(coordinate)
        pred = self.outer_most_non_linearity(pred)
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
            self.last_train_coords = batch['coord'].detach()

            self.optimizer.zero_grad()
            loss, loss_log = self.train_loss_fn(model, batch['coord'], batch['rgb']); total_loss.add(loss.item())
            loss.backward()
            self.optimizer.step()
            pbar.set_description(f'train: loss={loss_log:.4f}')
        return total_loss

    def valid_step(self, model, batch, patch_size=[128,128]):
        model.eval()
        valid_psnr = utils.utils.Averager()
        valid_ber  = utils.utils.Averager()
        valid_ssim = utils.utils.Averager()
        valid_lpips = utils.utils.Averager()

        n_inf = 0 
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

            psnr         = utils.metrics.calc_psnr(full_output, full_gt)
            ber, ber_lst = utils.metrics.calculate_ber(full_output, full_gt)
            ssim         = utils.metrics.calc_ssim(full_output, full_gt)
            lpips_val    = self._compute_lpips(full_output[0], full_gt[0])

            if ber == 0:
                n_inf = n_inf + 1
                psnr = 0          # set psnr 0 for stop indicator 

            valid_ber.add(ber)
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
        
        return valid_psnr, valid_ber, valid_ssim, valid_lpips, ber_lst, n_inf, full_output[0], full_gt[0], last_coord

    def train_eval_step(self, model, patch_size=[128,128]):
        model.eval()
        train_psnr = utils.utils.Averager()
        pbar = tqdm(self.loaders['train'], leave=False, desc='train_eval')
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
            train_psnr.add(psnr)
        return train_psnr

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
        latest_eval_psnr = None
        patch_size = self.cfg.datasets.train.wrapper.args.sr_config.patch_size
        for epoch in range(start_epoch, max_epoch + 1, iteration_per_epoch):
            self.log_buffer = [f'Epoch {epoch}']

            # [1] Train
            train_loss = self.train_step(model, self.loaders['train'])
            self.log_buffer.append(f'train: loss={train_loss.item():.4f}')
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # [2] Evaluation 
            if epoch % eval_epoch == 0:
                valid_psnr, valid_ber, valid_ssim, valid_lpips, ber_lst, n_inf, img, gt_img, valid_coord = self.valid_step(
                    model, self.loaders['valid'], patch_size
                )
                train_psnr = self.train_eval_step(model, patch_size)
                latest_eval_psnr = valid_psnr.item()

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
                    f'ber={valid_ber.item():.4f}, '
                    f'ssim={valid_ssim.item():.4f}, '
                    f'lpips={valid_lpips.item():.4f}, '
                    f'n_inf={n_inf}'
                )
                if valid_psnr.item() > best_metric:
                    best_metric = valid_psnr.item()
                    self.save_ckpt(model, 'finite_best.pth')
                    save_image(img, os.path.join(self.cfg._env.save_dir, 'finite_best.png'))
                if valid_psnr.item() == 0: 
                    self.save_ckpt(model, 'infinite.pth')
                    save_image(img, os.path.join(self.cfg._env.save_dir, 'infinite.png'))

                if self.enable_wandb:
                    log_dict = {
                        'epoch': epoch,
                        'loss': train_loss.item(),
                        'full_psnr': valid_psnr.item(),
                        'ssim': valid_ssim.item(),
                        'lpips': valid_lpips.item(),
                        'train_psnr': train_psnr.item(),
                        'test_psnr': 0.0,
                        'ber': valid_ber.item(),
                    }
                    if len(self.optimizer.param_groups) > 1:
                        log_dict['learning_rate_muon'] = self.optimizer.param_groups[0]['lr']
                        log_dict['learning_rate_aux'] = self.optimizer.param_groups[1]['lr']
                    else:
                        log_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']
                    for bit_idx, ber in enumerate(ber_lst):
                        log_dict[f'ber/bit_{bit_idx}'] = ber

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
            if latest_eval_psnr == 0:
                self.training_stop_callback()
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
        self._init_frequency_tracking(model)
        self.make_optimizer(model, self.cfg.optimizer, self.cfg.optimizer.schedulers)
        self.register_loss_fn(self.cfg.loss)
        self.run_train(model)
