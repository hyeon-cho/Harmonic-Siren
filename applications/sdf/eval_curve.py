"""Batch-evaluate all checkpoints of one or more runs into a Chamfer curve CSV.

Single persistent process (no per-checkpoint startup cost); resolution 128 by
default for curves (use eval_chamfer.py at 256 for headline numbers).

Usage:
    python eval_curve.py --runs logs/expA logs/expB --model_type gam ... --out curves.csv
"""
import argparse
import csv
import glob
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

import modules
import sdf_meshing
from eval_chamfer import SDFDecoder, normalize_like_dataio, chamfer


def build_model(args):
    if args.model_type == 'resin':
        return modules.SingleBVPNet(out_features=1, type='resin', in_features=3,
                                    hidden_features=args.hidden_features,
                                    num_hidden_layers=args.num_hidden_layers,
                                    recurrent_steps=args.recurrent_steps,
                                    recurrent_idx=args.recurrent_idx)
    return modules.SingleBVPNet(type='sine', in_features=3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--model_type', default='resin', choices=['resin', 'sine'])
    p.add_argument('--hidden_features', type=int, default=500)
    p.add_argument('--num_hidden_layers', type=int, default=1)
    p.add_argument('--recurrent_steps', type=int, default=3)
    p.add_argument('--recurrent_idx', type=int, default=None)
    p.add_argument('--point_cloud_path', default='point_cloud/armadillo.xyz')
    p.add_argument('--resolution', type=int, default=128)
    p.add_argument('--n_sample', type=int, default=60000)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    gt = normalize_like_dataio(np.genfromtxt(args.point_cloud_path)[:, :3])
    gt_tree_src = gt  # chamfer() builds trees internally

    model = build_model(args).cuda()
    rows = []
    for run in args.runs:
        ckpts = sorted(glob.glob(os.path.join(run, 'checkpoints', 'model_epoch_*.pth')))
        final = os.path.join(run, 'checkpoints', 'model_final.pth')
        if os.path.exists(final):
            ckpts.append(final)
        for ck in ckpts:
            m = re.search(r'model_epoch_(\d+)\.pth', ck)
            epoch = int(m.group(1)) if m else -1  # -1 = final
            model.load_state_dict(torch.load(ck, map_location='cpu'))
            tmp = os.path.join(run, 'eval', f'_curve_tmp')
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            try:
                sdf_meshing.create_mesh(SDFDecoder(model), tmp, N=args.resolution)
                mesh = trimesh.load(tmp + '.ply', force='mesh')
                if len(mesh.faces) == 0:
                    cd, comp = float('nan'), 0
                else:
                    pts, _ = trimesh.sample.sample_surface(mesh, args.n_sample)
                    cd = chamfer(np.asarray(pts), gt_tree_src)
                    comp = len(mesh.split(only_watertight=False))
            except Exception as e:
                print(f'{run} ep{epoch}: FAILED {e}')
                cd, comp = float('nan'), -1
            rows.append({'run': os.path.basename(run.rstrip('/')), 'epoch': epoch,
                         'chamfer_x1e6': cd * 1e6, 'components': comp})
            print(f"{rows[-1]['run']} ep{epoch}: CD={cd*1e6:.2f} comp={comp}", flush=True)

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'epoch', 'chamfer_x1e6', 'components'])
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {args.out}')


if __name__ == '__main__':
    main()
