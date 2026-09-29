# ReSIN: paper experiments

Code for **Recurrent Sinusoidal INRs for Efficient High-Fidelity Representation** (ECCV 2026; [arXiv:2607.21485](https://arxiv.org/abs/2607.21485)). This release contains the paper's image fitting, Kodak ×2 super-resolution, recurrence and coding studies, frequency analysis, and Armadillo SDF experiment.

## Contents

| Path | Experiment |
| --- | --- |
| `run.py`, `models/`, `trainers/`, `dataconf/`, `utils/`, `cfgs/` | Single-image 2D fitting |
| `float_reproduction/` | Continuous RGB runs for the paper's input-frequency and recurrent-step tables |
| `scripts/` | Per-image benchmark launcher, ×2 evaluation, coding comparison, spectral analysis, wall-clock extraction |
| `applications/sdf/` | SIREN and recurrent-SIREN Armadillo SDF training and evaluation |
| `paper_results/` | Source CSVs and reconstructions used for the input-frequency and recurrent-step tables |

The packaged data include a Set5 butterfly demo image and the Armadillo point cloud. For the full image benchmarks, obtain Set5, Kodak24, DIV2K, and FFHQ separately and pass an image directory to `--data`. Trained checkpoints and generated logs are intentionally outside this source archive.

## Setup

Use Python 3.10 and install a CUDA-compatible PyTorch build first, then:

```bash
pip install -r requirements.txt
```

Run all commands from the directory shown in each example. A CUDA GPU is required by the original training code.

## 2D fitting and Kodak ×2 super-resolution

The demo fits the Set5 butterfly. `run.py` expects a directory containing one image. Its built-in validation measures reconstruction on training-resolution coordinates. The ×2 result requires `scripts/evaluate_sr.py`, which queries the saved model at the upsampled coordinates and compares it with the original high-resolution image.

```bash
python run.py --cfg cfgs/General/ReSIN.yaml --gpu 0
python scripts/run_image_benchmark.py --input /path/to/Kodak24 --gpu 0
python scripts/evaluate_sr.py \
  --checkpoint save/paper_images/0000_IMAGE_STEM/infinite.pth \
  --hr-image /path/to/Kodak24/IMAGE_STEM.png \
  --output save/paper_images/0000_IMAGE_STEM/sr_x2.png --lpips
```

The benchmark launcher makes one temporary image directory per input and fits one model per image. Replace `IMAGE_STEM` with the actual result directory and file name. The ×2 evaluator applies the training configuration's center crop and resizes the reference to `512×512`; use the same high-resolution source image as the fitting run. If exact reconstruction was not reached, pass `finite_best.pth` instead of `infinite.pth`.

The `ReSIN_H*.yaml` files change the hidden width. `REC_H768.yaml` is the continuous RGB recurrent comparison. For Gray-code versus natural-binary supervision:

```bash
bash scripts/ablate_coding.sh 0
```

Training outputs are written under `save/`, which is ignored by Git.

## Input frequency and recurrent-step tables

These tables used the continuous RGB training lineage and a separate trainer. Run it from `float_reproduction/`; its configs and demo image are self-contained. The recorded table values are in `paper_results/`.

```bash
cd float_reproduction
bash run_table2.sh 0
bash run_table1.sh 0
```

The `truth_R*.yaml` configs reproduce the recurrent-step setting: 768 hidden units, three RGB outputs, `omega_0={first: 256, hidden: 64}`, and five choices of recurrent steps. The `truth_omega*.yaml` configs vary the input frequency. In the original Table 2 logs, one recorded epoch corresponds to ten optimizer updates; the paper's reported step 500 is recorded epoch 500.

## Spectral analysis

After training a ReSIN model, pass its checkpoint to the depthwise spectrum script:

```bash
python scripts/spectral_support_vs_depth.py --ckpt save/EXPERIMENT/infinite.pth --grid 256 --thresh-db -40
```

`spectral_support_vs_depth_general.py` compares registered architectures on a coordinate grid. `wallclock_from_log.py` extracts time-to-quality points from a training log.

## Armadillo SDF

From `applications/sdf/`, run the paper's 2,000-epoch SIREN and recurrent-sinusoidal training matrix:

```bash
cd applications/sdf
bash run_paper_matrix.sh
```

The training and evaluation scripts, point cloud, and summary CSVs are in that directory. See its [experiment notes](applications/sdf/README.md). The SDF code adapts the [SIREN implementation](https://github.com/vsitzmann/siren) and includes its MIT license; the minimal `torchmeta/modules` subset is from [pytorch-meta](https://github.com/tristandeleu/pytorch-meta) with its license retained.
