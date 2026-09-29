"""Validate the depth->spectral-support claim of recurrent SineLayer INRs.

Forward a 2D coord grid through a trained MININR-ol, capture the activations
after each recurrent application, FFT them, and report two summaries per
depth ell:

  1. spectral support |S_ell|: number of frequency bins whose mean magnitude
     across hidden channels exceeds a threshold (default: -40 dB below max).
  2. radially-binned magnitude profile (input to a plot if desired).

Theory predicts (Jacobi-Anger):
   |S_ell+1| >= |S_ell|, with new lines at integer combinations of S_ell.

Usage from repo root:
  python scripts/spectral_support_vs_depth.py \
      --ckpt save/GAM_H768_768_R5_adamw_None_1.5e-04_0002/infinite.pth \
      --grid 256 --thresh-db -40
"""
import argparse
import json
import math
import os
import sys
import torch
import numpy as np

# Make `models`, `utils`, etc importable when invoked from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models  # registers MININR-ol
from utils.geometry import make_coord


def load_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    spec = ckpt['model']
    args = dict(spec['args'])
    sd = spec['sd']
    model = models.make({'name': spec['name'], 'args': args}, load_sd=False)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, spec


@torch.no_grad()
def capture_activations(model, coord, max_depth=None):
    """Walk the MININR-ol forward, capturing h_l = recurrnet_net(...) for l=0..L.

    h_0 = in_layer(coord)              shape (N, H)
    h_l = recurrnet_net(h_{l-1})       for l=1..L
    """
    L = max_depth if max_depth is not None else model.recurrent_steps
    activations = []
    h = model.in_layer(coord)
    activations.append(h)            # h_0 (encoder output)
    for _ in range(L):
        h = model.recurrnet_net(h)
        activations.append(h)        # h_1, h_2, ..., h_L
    return activations


def fft2_magnitude_db(x_hw, eps=1e-12):
    """Return log10 of mean over channels of |FFT2|, normalized to peak."""
    # x_hw: (H, W, C)
    f = torch.fft.fftshift(torch.fft.fft2(x_hw, dim=(0, 1)), dim=(0, 1))
    mag = torch.abs(f).mean(dim=-1)               # (H, W)
    db  = 20.0 * torch.log10(mag / (mag.max() + eps) + eps)
    return db, mag


def spectral_support(db, thresh_db):
    return int((db > thresh_db).sum().item())


def radial_average(mag):
    """Average magnitude in radial frequency bins. mag is (H, W) post-fftshift."""
    H, W = mag.shape
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H) - cy, torch.arange(W) - cx, indexing='ij')
    rr = torch.sqrt((yy.float()**2 + xx.float()**2))
    n_bins = min(H, W) // 2
    profile = torch.zeros(n_bins)
    counts = torch.zeros(n_bins)
    rr_int = rr.long().clamp(max=n_bins - 1).flatten()
    flat = mag.flatten()
    profile.scatter_add_(0, rr_int, flat.cpu())
    counts.scatter_add_(0, rr_int, torch.ones_like(flat).cpu())
    return (profile / counts.clamp(min=1)).numpy()


def init_model(name, args_dict, seed=0):
    """Build a fresh, untrained model with the given args."""
    torch.manual_seed(seed)
    return models.make({'name': name, 'args': dict(args_dict)}, load_sd=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=None,
                   help='If given, load weights from this checkpoint. Otherwise use --untrained.')
    p.add_argument('--untrained', action='store_true',
                   help='Probe a freshly-initialized untrained MININR-ol with --hidden / --omega-first / --omega-hidden')
    p.add_argument('--hidden', type=int, default=768)
    p.add_argument('--omega-first', type=float, default=256.0)
    p.add_argument('--omega-hidden', type=float, default=45.0)
    p.add_argument('--in-channels', type=int, default=2)
    p.add_argument('--out-channels', type=int, default=24)
    p.add_argument('--grid', type=int, default=256)
    p.add_argument('--max-depth', type=int, default=8,
                   help='Number of recurrent applications to probe')
    p.add_argument('--thresh-db', type=float, default=-40.0)
    p.add_argument('--coord-range', type=float, nargs=2, default=[0.0, 1.0])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/spectral_vs_depth')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.untrained:
        spec = {'name': 'MININR-ol',
                'args': {
                    'in_channels': args.in_channels,
                    'hidden_channels': args.hidden,
                    'out_channels': args.out_channels,
                    'bias': False,
                    'recurrent_steps': args.max_depth,
                    'omega_0': {'first': args.omega_first, 'hidden': args.omega_hidden},
                }}
        model = init_model(spec['name'], spec['args'], seed=args.seed)
    else:
        if not args.ckpt:
            raise SystemExit('Either --ckpt PATH or --untrained must be given.')
        model, spec = load_ckpt(args.ckpt)
    model = model.to(device)
    H = args.grid
    coord_2d = make_coord([H, H],
                          ranges=[args.coord_range, args.coord_range])
    coord = coord_2d.to(device)            # (HW, 2)

    activations = capture_activations(model, coord, max_depth=args.max_depth)

    rows = []
    for l, h in enumerate(activations):
        # h: (HW, C) -> (H, H, C)
        x = h.reshape(H, H, -1)
        db, mag = fft2_magnitude_db(x)
        supp = spectral_support(db, args.thresh_db)
        radial = radial_average(mag)
        # Normalize radial profile to peak (1.0 at DC)
        radial_norm = (radial / (radial[0] + 1e-12)).tolist()
        rows.append({
            'depth': l,
            'channels': int(x.shape[-1]),
            'spectral_support': supp,
            'support_fraction': supp / float(H * H),
            'radial_profile_normalized_first8': radial_norm[:8],
        })

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    out_json = args.out + '.json'
    with open(out_json, 'w') as f:
        json.dump({
            'ckpt': args.ckpt,
            'model_name': spec['name'],
            'grid': H,
            'thresh_db': args.thresh_db,
            'rows': rows,
        }, f, indent=2)
    print(f'Wrote {out_json}')

    # Print headline table
    print()
    src = args.ckpt if args.ckpt else f'untrained MININR-ol (H={args.hidden}, ω={args.omega_first}/{args.omega_hidden})'
    print(f'## Spectral support vs recurrent depth')
    print(f'(source: {src}, grid: {H}x{H}, threshold: {args.thresh_db} dB below peak)')
    print()
    print('| depth ell | hidden channels | spectral_support |S_ell| | support fraction |')
    print('|---:|---:|---:|---:|')
    for r in rows:
        frac_pct = 100 * r['support_fraction']
        print(f'| {r["depth"]} | {r["channels"]} | {r["spectral_support"]:>6} | {frac_pct:.2f}% |')


if __name__ == '__main__':
    main()
