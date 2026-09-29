"""Spectral support vs depth across architectures.

Walks any registered model layer-by-layer (or recurrent-step-by-step) using
either explicit traversal (for known recurrent classes) or forward hooks
(for nn.Sequential-style baselines), then computes
    |S_l| = |{(u,v) : 20 log10(M_l[u,v] / max M_l) > tau}|
where M_l is the channel-averaged magnitude of the 2D FFT of the l-th
intermediate activation reshaped to (H, H, C).

Usage from repo root, e.g.:
    python scripts/spectral_support_vs_depth_general.py \
        --arch FINER --hidden 345 --omega-first 30 --omega-hidden 30 --bias true \
        --grid 256 --thresh-db -40
"""
import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models  # registers all archs incl. baselines
from utils.geometry import make_coord


def fft2_db(x_hw, eps=1e-12):
    """x_hw: (H, W, C). Returns db (H, W) and raw mag (H, W)."""
    x_hw = x_hw.float() if x_hw.dtype not in (torch.float32, torch.float64) else x_hw
    f = torch.fft.fftshift(torch.fft.fft2(x_hw, dim=(0, 1)), dim=(0, 1))
    mag = torch.abs(f).mean(dim=-1)
    db = 20.0 * torch.log10(mag / (mag.max() + eps) + eps)
    return db, mag


def support_count(db, thresh_db):
    return int((db > thresh_db).sum().item())


def real_view(t):
    """If t is complex, take real part; otherwise return as-is."""
    return t.real if torch.is_complex(t) else t


###############################################################################
# Arch-specific traversal: captures *recurrent applications* for our weight-
# tied models, and *sequential layers* for vanilla baselines.
###############################################################################
@torch.no_grad()
def probe_activations(model, coord, max_depth=8):
    """Returns a list of (label, tensor (HW, C)) pairs.

    The first element is "depth-0" (encoder output, before any hidden
    nonlinearity beyond the first), and subsequent elements are after each
    additional layer / recurrent application up to max_depth.
    """
    name = type(model).__name__

    # ---- weight-tied recurrent INRs ----
    if name == 'OursINR':                # MININR-ol used by ReSIN/ReSINW/REC (legacy save dirs use GAM prefixes)
        acts = [('h0', model.in_layer(coord))]
        h = acts[0][1]
        for i in range(max_depth):
            h = model.recurrnet_net(h)
            acts.append((f'h{i+1}', h))
        return acts

    if name == 'RecurrentSIREN':
        acts = [('h0', model.input_layer(coord))]
        h = acts[0][1]
        for i in range(max_depth):
            h = model.recurrent_layer(h)
            acts.append((f'h{i+1}', h))
        return acts

    if name == 'FINER_Rec':
        acts = [('h0', model.inp_layer(coord))]
        h = acts[0][1]
        for i in range(max_depth):
            h = model.rec_layer(h)
            acts.append((f'h{i+1}', h))
        return acts

    if name == 'Gauss_Rec':
        acts = [('h0', model.input_layer(coord))]
        h = acts[0][1]
        for i in range(max_depth):
            h = model.rec_layer(h)
            acts.append((f'h{i+1}', h))
        return acts

    if name == 'PEMLP_Rec':
        encoded = model.encoding(coord)
        h0 = model.relu(model.input_linear(encoded))
        acts = [('h0', h0)]
        h = h0
        for i in range(max_depth):
            h = model.relu(model.rec_linear(h))
            acts.append((f'h{i+1}', h))
        return acts

    if name == 'WIRE_Rec':
        acts = [('h0', model.input_layer(coord))]
        h = acts[0][1]
        for i in range(max_depth):
            h = model.rec_layer(h)
            acts.append((f'h{i+1}', h))
        return acts

    # ---- iMFN / iSIREN / iGabor / ImpSq: DEQ over MFN. Depth = number of inner-cell applications. ----
    if name in ('ImpSq', 'iMFN', 'iSIREN', 'iGabor'):
        # z_0 = zeros; z_{l+1} = f(x, z_l) where f is the MFN inner cell.
        # We capture z_0 = MFN(x, zeros) (= init_filter(x)) as h0, then iterate.
        z = torch.zeros(*coord.shape[:-1], model.interm_channels,
                        device=coord.device, dtype=coord.dtype)
        acts = []
        for i in range(max_depth + 1):
            z = model.f(coord, z)
            acts.append((f'h{i}', z))
        return acts

    # ---- MFN / BACON: multiplicative filter structure (filters[i](x) * linear[i-1](out)) ----
    if name in ('FourierNet', 'GaborNet', 'BACON'):
        acts = []
        out = model.filters[0](coord)
        acts.append(('h0', out))
        for i in range(1, len(model.filters)):
            out = model.filters[i](coord) * model.linear[i - 1](out)
            acts.append((f'h{i}', out))
            if i >= max_depth:
                break
        return acts

    # ---- non-recurrent baselines: walk nn.Sequential layers ----
    # SIREN, FINER, Gauss, WIRE, ReLU_MLP have self.net = Sequential of layers
    # PEMLP also has self.encoding then self.net
    if name == 'PEMLP':
        x = model.encoding(coord)
    else:
        x = coord
    acts = []
    if hasattr(model, 'net') and isinstance(model.net, nn.Sequential):
        seq = model.net
    else:
        raise RuntimeError(f'no traversal known for arch={name}')
    # Capture after each "block" — we treat a block as the chunk that ends
    # with a non-linearity (or the final linear). For Sequential([Linear,ReLU,
    # Linear,ReLU,..., Linear]) this means capture after each ReLU. For
    # Sequential([SineLayer, SineLayer, ..., Linear]) it's after each SineLayer.
    h = x
    captured = 0
    for k, layer in enumerate(seq):
        h = layer(h)
        is_last = (k == len(seq) - 1)
        # Capture after each layer-with-nonlinearity (skip pure Linear unless final)
        is_nonlinearity_block = isinstance(layer, (nn.ReLU,)) or 'Layer' in type(layer).__name__
        if is_nonlinearity_block or is_last:
            acts.append((f'h{captured}', real_view(h)))
            captured += 1
            if captured > max_depth + 1:
                break
    return acts


def init_model(arch, args_dict, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return models.make({'name': arch, 'args': dict(args_dict)}, load_sd=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', required=True)
    p.add_argument('--hidden', type=int, required=True)
    p.add_argument('--out-channels', type=int, default=3)
    p.add_argument('--in-channels', type=int, default=2)
    p.add_argument('--omega-first', type=float, default=None)
    p.add_argument('--omega-hidden', type=float, default=None)
    p.add_argument('--bias', type=str, default=None,
                   help='true/false; auto-picked if None')
    p.add_argument('--num-layers', type=int, default=None,
                   help='for SIREN/RecurrentSIREN/PEMLP/PEMLP_Rec/ReLU_MLP')
    p.add_argument('--hidden-layers', type=int, default=None,
                   help='for FINER/FINER_Rec/Gauss/Gauss_Rec/WIRE/WIRE_Rec')
    p.add_argument('--scale', type=float, default=None,
                   help='for Gauss/Gauss_Rec/WIRE/WIRE_Rec')
    p.add_argument('--n-freqs', type=int, default=10,
                   help='for PEMLP/PEMLP_Rec')
    p.add_argument('--grid', type=int, default=256)
    p.add_argument('--max-depth', type=int, default=8)
    p.add_argument('--thresh-db', type=float, default=-40.0)
    p.add_argument('--coord-range', type=float, nargs=2, default=[0.0, 1.0])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', default='2')
    p.add_argument('--ckpt', default=None,
                   help='If set, load weights from this checkpoint (overrides untrained init)')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build constructor args.
    bias = (args.bias.lower() == 'true') if args.bias is not None else None

    arch_args = dict(in_channels=args.in_channels,
                     hidden_channels=args.hidden,
                     out_channels=args.out_channels)
    if args.arch == 'OursINR' or args.arch == 'MININR-ol':
        arch_args.update(bias=False if bias is None else bias,
                         recurrent_steps=args.max_depth,
                         omega_0={'first': args.omega_first or 256.0,
                                  'hidden': args.omega_hidden or 45.0})
        arch_name = 'MININR-ol'
    elif args.arch == 'SIREN':
        arch_args.update(bias=True if bias is None else bias,
                         num_layers=args.num_layers or 5,
                         outer_most_linear=True,
                         omega_0={'first': args.omega_first or 30.0,
                                  'hidden': args.omega_hidden or 30.0})
        arch_name = 'SIREN'
    elif args.arch == 'RecurrentSIREN':
        arch_args.update(bias=False if bias is None else bias,
                         num_layers=args.num_layers or 5,
                         outer_most_linear=True,
                         omega_0={'first': args.omega_first or 256.0,
                                  'hidden': args.omega_hidden or 45.0})
        arch_name = 'RecurrentSIREN'
    elif args.arch == 'FINER':
        arch_args.update(bias=True if bias is None else bias,
                         hidden_layers=args.hidden_layers or 5,
                         omega_0={'first': args.omega_first or 30.0,
                                  'hidden': args.omega_hidden or 30.0},
                         first_bias_scale=None,
                         scale_req_grad=False)
        arch_name = 'FINER'
    elif args.arch == 'FINER_Rec':
        arch_args.update(bias=False if bias is None else bias,
                         hidden_layers=args.hidden_layers or 5,
                         omega_0={'first': args.omega_first or 256.0,
                                  'hidden': args.omega_hidden or 45.0},
                         first_bias_scale=None,
                         scale_req_grad=False)
        arch_name = 'FINER_Rec'
    elif args.arch == 'Gauss':
        arch_args.update(hidden_layers=args.hidden_layers or 3,
                         scale=args.scale or 30.0)
        arch_name = 'Gauss'
    elif args.arch == 'Gauss_Rec':
        arch_args.update(hidden_layers=args.hidden_layers or 5,
                         scale=args.scale or 30.0)
        arch_name = 'Gauss_Rec'
    elif args.arch == 'WIRE':
        arch_args.update(hidden_layers=args.hidden_layers or 3,
                         omega_0={'first': args.omega_first or 20.0,
                                  'hidden': args.omega_hidden or 20.0},
                         scale=args.scale or 10.0)
        arch_name = 'WIRE'
    elif args.arch == 'ReLU_MLP':
        arch_args.update(num_layers=args.num_layers or 5)
        arch_name = 'ReLU_MLP'
    elif args.arch == 'PEMLP':
        arch_args.update(num_layers=args.num_layers or 5,
                         N_freqs=args.n_freqs)
        arch_name = 'PEMLP'
    elif args.arch == 'PEMLP_Rec':
        arch_args.update(num_layers=args.num_layers or 5,
                         N_freqs=args.n_freqs)
        arch_name = 'PEMLP_Rec'
    elif args.arch == 'FourierNet':
        arch_args.update(n_layers=args.num_layers or 3,
                         input_scale=args.omega_first or 256.0,
                         weight_scale=args.scale or 1.0,
                         bias=True if bias is None else bias)
        arch_name = 'FourierNet'
    elif args.arch == 'GaborNet':
        arch_args.update(n_layers=args.num_layers or 3,
                         input_scale=args.omega_first or 256.0,
                         weight_scale=args.scale or 1.0,
                         alpha=6.0, beta=1.0,
                         bias=True if bias is None else bias)
        arch_name = 'GaborNet'
    elif args.arch == 'BACON':
        arch_args.update(hidden_layers=args.hidden_layers or 3,
                         weight_scale=args.scale or 1.0,
                         bias=True if bias is None else bias,
                         frequency=(args.omega_first or 128.0,
                                    args.omega_first or 128.0))
        arch_name = 'BACON'
    elif args.arch in ('ImpSq', 'iMFN', 'iSIREN', 'iGabor'):
        arch_args.update(n_layers=args.num_layers or 4,
                         input_scale=args.omega_first or 256.0,
                         max_iter=30, tol=1e-3, output_act=False)
        arch_name = args.arch
    else:
        raise SystemExit(f'unknown arch {args.arch}')

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        spec = ckpt['model']
        model = models.make({'name': spec['name'], 'args': dict(spec['args'])}, load_sd=False)
        model.load_state_dict(spec['sd'], strict=True)
        arch_name = spec['name']
        print(f'Loaded checkpoint {args.ckpt} for arch={arch_name}')
    else:
        model = init_model(arch_name, arch_args, seed=args.seed)
    model = model.to(device).eval()
    H = args.grid
    coord = make_coord([H, H], ranges=[args.coord_range, args.coord_range]).to(device)

    acts = probe_activations(model, coord, max_depth=args.max_depth)

    def radial_profile(mag):
        H_, W_ = mag.shape
        cy, cx = H_ // 2, W_ // 2
        yy, xx = torch.meshgrid(torch.arange(H_) - cy, torch.arange(W_) - cx, indexing='ij')
        rr = torch.sqrt(yy.float()**2 + xx.float()**2)
        n_bins = min(H_, W_) // 2
        prof = torch.zeros(n_bins); cnt = torch.zeros(n_bins)
        rrf = rr.long().clamp(max=n_bins - 1).flatten()
        flat = mag.cpu().flatten()
        prof.scatter_add_(0, rrf, flat)
        cnt.scatter_add_(0, rrf, torch.ones_like(flat))
        return (prof / cnt.clamp(min=1)).numpy().tolist()

    rows = []
    for ell, (lbl, h) in enumerate(acts):
        x = h.reshape(H, H, -1)
        db, mag = fft2_db(x)
        supp = support_count(db, args.thresh_db)
        radial = radial_profile(mag)
        rows.append({
            'depth': ell,
            'label': lbl,
            'channels': int(x.shape[-1]),
            'spectral_support': supp,
            'support_fraction': supp / float(H * H),
            'radial_profile': radial,
        })

    out = args.out or f'results/spec_{arch_name}_H{args.hidden}'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    payload = {'arch': arch_name, 'hidden': args.hidden,
               'omega_first': args.omega_first, 'omega_hidden': args.omega_hidden,
               'scale': args.scale, 'bias': args.bias, 'grid': H,
               'thresh_db': args.thresh_db, 'rows': rows}
    with open(out + '.json', 'w') as f: json.dump(payload, f, indent=2)
    print(f'\n--- {arch_name}  H={args.hidden}'
          f'{f"  ω={args.omega_first}/{args.omega_hidden}" if args.omega_first else ""}'
          f'{f"  scale={args.scale}" if args.scale else ""} ---')
    print('| depth | label | channels | |S| | fraction |')
    print('|--:|--|--:|--:|--:|')
    for r in rows:
        print(f'| {r["depth"]} | {r["label"]} | {r["channels"]} | {r["spectral_support"]} | {r["support_fraction"]*100:.2f}% |')
    print(f'(saved {out}.json)')


if __name__ == '__main__':
    main()
