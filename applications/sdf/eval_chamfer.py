"""Extract a mesh from a trained SDF checkpoint, compute the Chamfer distance to the
ground-truth point cloud, and save a quick 3-view visualization.

Usage:
    python eval_chamfer.py --checkpoint logs/<exp>/checkpoints/model_final.pth \
        --model_type resin --hidden_features 500 --num_hidden_layers 1 --recurrent_steps 3 \
        --point_cloud_path point_cloud/armadillo.xyz --out logs/<exp>/eval
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

import modules
import sdf_meshing


class SDFDecoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, coords):
        return self.model({'coords': coords})['model_out']


def normalize_like_dataio(coords):
    """Replicate dataio.PointCloud normalization (keep_aspect_ratio=True)."""
    coords = coords - coords.mean(axis=0, keepdims=True)
    cmax, cmin = coords.max(), coords.min()
    coords = (coords - cmin) / (cmax - cmin)
    return (coords - 0.5) * 2.0


def chamfer(a, b):
    """Symmetric Chamfer distance (mean squared nn distance, both directions)."""
    d_ab = cKDTree(b).query(a, k=1)[0]
    d_ba = cKDTree(a).query(b, k=1)[0]
    return (d_ab ** 2).mean() + (d_ba ** 2).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model_type', default='resin', choices=['resin', 'sine'])
    p.add_argument('--hidden_features', type=int, default=500)
    p.add_argument('--num_hidden_layers', type=int, default=1)
    p.add_argument('--recurrent_steps', type=int, default=3)
    p.add_argument('--recurrent_idx', type=int, default=None)
    p.add_argument('--point_cloud_path', default='point_cloud/armadillo.xyz')
    p.add_argument('--resolution', type=int, default=256)
    p.add_argument('--n_sample', type=int, default=100000)
    p.add_argument('--out', required=True, help='output prefix (writes <out>.ply/.png/.txt)')
    args = p.parse_args()

    if args.model_type == 'resin':
        model = modules.SingleBVPNet(out_features=1, type='resin', in_features=3,
                                     hidden_features=args.hidden_features,
                                     num_hidden_layers=args.num_hidden_layers,
                                     recurrent_steps=args.recurrent_steps,
                                     recurrent_idx=args.recurrent_idx)
    else:
        model = modules.SingleBVPNet(type='sine', in_features=3)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.cuda()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    sdf_meshing.create_mesh(SDFDecoder(model), args.out, N=args.resolution)

    mesh = trimesh.load(args.out + '.ply', force='mesh')
    mesh_pts, _ = trimesh.sample.sample_surface(mesh, args.n_sample)
    gt = normalize_like_dataio(np.genfromtxt(args.point_cloud_path)[:, :3])

    cd = chamfer(np.asarray(mesh_pts), gt)
    n_comp = len(mesh.split(only_watertight=False))
    report = (f"checkpoint={args.checkpoint}\n"
              f"vertices={len(mesh.vertices)} faces={len(mesh.faces)} components={n_comp}\n"
              f"chamfer(x1e6)={cd * 1e6:.2f}\n")
    print(report)
    with open(args.out + '.txt', 'w') as f:
        f.write(report)

    # quick 3-view scatter for visual inspection
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sub = np.asarray(mesh_pts)[::5]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (i, j, name) in zip(axes, [(0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ')]):
        ax.scatter(sub[:, i], sub[:, j], s=0.1, c=sub[:, 2 if name != 'YZ' else 0],
                   cmap='viridis')
        ax.set_title(name); ax.set_aspect('equal'); ax.axis('off')
    fig.suptitle(os.path.basename(args.out))
    fig.savefig(args.out + '.png', dpi=110, bbox_inches='tight')
    print(f"saved {args.out}.ply / .png / .txt")


if __name__ == '__main__':
    main()
