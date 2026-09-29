"""Query a trained image INR on its ×2 coordinate grid and compare to an HR image."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models
from models.activations import AperiodicSine
from utils.discretization import BiTransform, GrayTransform
from utils.geometry import make_coord
from utils.metrics import calc_lpips, calc_psnr, calc_ssim


def make_sr_coords(cfg, size, scale):
    args = cfg['datasets']['train']['wrapper']['args']
    coord_cfg = args['coord_config']
    coords = make_coord((size, size), coord_cfg['range'])
    coords[:, 0] /= coord_cfg['range_division'][0]
    coords[:, 1] /= coord_cfg['range_division'][1]
    return rearrange(
        F.interpolate(rearrange(coords, '(h w) xy -> () xy h w', h=size),
                      scale_factor=scale, mode='bilinear').squeeze(0),
        'xy h w -> (h w) xy',
    )


def prepare_hr(path, target_size):
    image = Image.open(path).convert('RGB')
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def decode(pred, cfg, target_size):
    mode = cfg.get('trainer_mode', 'resin')
    outer = cfg.get('outer_non_linearity', 'asin')
    if outer == 'asin':
        pred = AperiodicSine()(pred)
    elif outer == 'tanh':
        pred = torch.tanh(pred) / 2
    elif outer != 'linear':
        raise ValueError(f'unknown outer_non_linearity: {outer}')

    if mode == 'rec':
        return rearrange(((pred + 1) / 2).clamp(0, 1), '(h w) c -> () c h w', h=target_size)
    if mode == 'mse':
        return rearrange(pred.clamp(0, 1), '(h w) c -> () c h w', h=target_size)

    transformer = cfg['datasets']['train']['wrapper']['args'].get('transformer', 'binary')
    if transformer == 'gray':
        coder = GrayTransform()
    elif transformer == 'binary':
        coder = BiTransform()
    else:
        raise ValueError(f'unknown transformer: {transformer}')
    bits = rearrange((pred > 0).float(), '(h w) (rgb bit) -> () bit rgb h w',
                     h=target_size, rgb=3)
    return coder.composition(bits).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--hr-image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=65536)
    parser.add_argument('--lpips', action='store_true', help='also calculate LPIPS')
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error('--chunk-size must be positive')

    cfg_path = args.checkpoint.with_name('cfg.yaml')
    cfg = yaml.safe_load(cfg_path.read_text())
    train_args = cfg['datasets']['train']['wrapper']['args']
    size = train_args['resize']
    scale = cfg['datasets']['valid']['wrapper']['args']['sr_config']['scale_factor']
    if scale != 2:
        parser.error(f'expected ×2 config, got scale_factor={scale}')
    target_size = size * scale

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.make(ckpt['model'], load_sd=True).to(device).eval()
    coords = make_sr_coords(cfg, size, scale)
    with torch.no_grad():
        pred = torch.cat([model(chunk.to(device)).cpu()
                          for chunk in coords.split(args.chunk_size)], dim=0)
    output = decode(pred, cfg, target_size)
    target = prepare_hr(args.hr_image, target_size)
    metrics = {
        'psnr': float(calc_psnr(output, target)),
        'ssim': float(calc_ssim(output, target)),
    }
    if args.lpips:
        metrics['lpips'] = float(calc_lpips(output, target))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image = (output[0].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(image).save(args.output)
    print(f'{args.output}: ' + ', '.join(f'{key}={value:.6f}' for key, value in metrics.items()))


if __name__ == '__main__':
    main()
