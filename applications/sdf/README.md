# Armadillo SDF experiment

This directory contains the SDF training and Chamfer evaluation code used with a one-hidden-layer, width-500 recurrent sinusoidal decoder unrolled three times, compared with SIREN over 2,000 epochs. The input point cloud is `point_cloud/armadillo.xyz`.

Install the repository's root `requirements.txt`, then run from this directory:

```bash
bash run_paper_matrix.sh
```

The script trains SIREN (`lr=1e-4`) and ReSIN (`lr=1e-4` and `5e-4`) for seeds 0, 1, and 2, writing checkpoints under `logs/`. For a trained checkpoint, use:

```bash
python eval_chamfer.py --checkpoint logs/mat_gam1e-4_s0/checkpoints/model_final.pth \
  --model_type resin --out logs/mat_gam1e-4_s0/eval/final
```

`eval_curve.py` evaluates checkpoint sequences. The archived summary and curves are in `paper_results/`. Run `python make_sdf_plots.py` to regenerate the Chamfer and component plots under `plots/` from those CSVs.

The SDF implementation is adapted from [SIREN](https://github.com/vsitzmann/siren) under the MIT license in `LICENSE`. The local `torchmeta/modules` subset comes from [pytorch-meta](https://github.com/tristandeleu/pytorch-meta); its MIT license is in `torchmeta/LICENSE`.
