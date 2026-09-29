"""Entry point for "Recurrent Sinusoidal INRs for Efficient High-Fidelity Representation".

Single-image overfitting with a weight-tied recurrent sinusoidal INR, trained either
with binarized (Gray-coded bipolar) supervision or as a continuous regressor.

    python run.py --cfg cfgs/General/ReSIN.yaml --gpu 0
"""


import warnings
warnings.filterwarnings("ignore")

import argparse
import os

import yaml
from omegaconf import OmegaConf

from trainers import trainers_dict

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/General/ReSIN.yaml')
    parser.add_argument('--opt', nargs='*', default=[])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume-mode', '-r', default='replace')

    parser.add_argument('--save-root', default='save')
    parser.add_argument('--name', '-n', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--wandb', '-w', action='store_true')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--data', default=None)
    parser.add_argument('--bypass', '-b', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args


def parse_cfg(cfg):
    if cfg.get('_base_') is not None:
        fnames = cfg.pop('_base_')
        if isinstance(fnames, str):
            fnames = [fnames]
        base_cfg = OmegaConf.merge(*[parse_cfg(OmegaConf.load(_)) for _ in fnames])
        cfg = OmegaConf.merge(base_cfg, cfg)
    return cfg


def apply_data_override(cfg, data):
    """Apply --data to both train/valid root_path. Must run regardless of --bypass."""
    if data is not None:
        print(f'\n========================================')
        print(f"Fitting Data is changed to {data}")
        print(f'========================================\n')
        data = data.rstrip('/')
        cfg.datasets.train.wrapper.args.dataset.args.root_path = data
        cfg.datasets.valid.wrapper.args.dataset.args.root_path = data

    assert cfg.datasets.train.wrapper.args.dataset.args.root_path == cfg.datasets.valid.wrapper.args.dataset.args.root_path, "Train and validation datasets must have the same root path."


def load_param_settings(cfg):
    hidden_neurons  = cfg.model.args.hidden_channels
    recurrent_steps = (cfg.model.args.get('recurrent_steps')
                       or cfg.model.args.get('num_layers')
                       or cfg.model.args.get('hidden_layers')
                       or 1)
    optimizer       = cfg.optimizer.name
    scheduler       = 'None' if cfg.optimizer.schedulers.name is None else cfg.optimizer.schedulers.name
    lr              = format(cfg.optimizer.args.lr, ".1e")
    dataset = cfg.datasets.train.wrapper.args.dataset.args.root_path.split('/')[-1]

    additional_string = f"_{hidden_neurons}_R{recurrent_steps}_{optimizer}_{scheduler}_{lr}_{dataset}"
    return additional_string


def make_cfg(args):
    cfg = parse_cfg(OmegaConf.load(args.cfg))
    for i in range(0, len(args.opt), 2):
        k, v = args.opt[i: i + 2]
        # yaml-parse the value so numbers/bools keep their type ('500' -> 500);
        # YAML treats bare exponents like '5e-4' as strings, so fall back to float.
        parsed = yaml.safe_load(v)
        if isinstance(parsed, str):
            try:
                parsed = float(parsed)
            except ValueError:
                pass
        OmegaConf.update(cfg, k, parsed)
    if args.seed is not None:
        cfg.random_seed = args.seed
    apply_data_override(cfg, args.data)

    env = OmegaConf.create()
    if args.name is None:
        exp_name = os.path.splitext(os.path.basename(args.cfg))[0]
    else:
        exp_name = args.name
    if args.tag is not None:
        exp_name += '_' + args.tag

    if not args.bypass:
        exp_name = exp_name + load_param_settings(cfg)

    env.exp_name = exp_name
    env.save_dir = os.path.join(args.save_root, exp_name)
    env.wandb = args.wandb
    env.resume_mode = args.resume_mode
    cfg._env = env

    return cfg

if __name__ == '__main__':
    args = make_args()
    cfg = make_cfg(args)
    trainer = trainers_dict[cfg.trainer](cfg)
    trainer.run()
