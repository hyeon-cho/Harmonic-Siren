"""Fit one INR per image in a flat image directory.

The original trainer fits one image at a time. This launcher makes temporary
single-image input directories and keeps each result under a distinct name.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='flat directory of images')
    parser.add_argument('--cfg', type=Path, default=Path('cfgs/General/ReSIN.yaml'))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-root', type=Path, default=Path('save/paper_images'))
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--dry-run', action='store_true', help='print commands without training')
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    source = args.input.resolve()
    if not source.is_dir():
        parser.error(f'not a directory: {source}')
    images = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        parser.error(f'no images found in {source}')
    if args.max_images is not None:
        if args.max_images < 1:
            parser.error('--max-images must be positive')
        images = images[:args.max_images]

    cfg = args.cfg if args.cfg.is_absolute() else repo / args.cfg
    if not cfg.is_file():
        parser.error(f'config not found: {cfg}')
    save_root = args.save_root.resolve()
    if not args.dry_run:
        save_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='resin_images_') as tmp:
        one_image_dir = Path(tmp) / 'image'
        one_image_dir.mkdir()
        for index, image in enumerate(images):
            link = one_image_dir / f'{index:04d}{image.suffix.lower()}'
            link.symlink_to(image)
            name = f'{index:04d}_{re.sub(r"[^A-Za-z0-9_.-]", "_", image.stem)}'
            result_dir = save_root / name
            if result_dir.exists():
                raise FileExistsError(f'result already exists: {result_dir}')
            print(f'[{index + 1}/{len(images)}] {image.name} -> {result_dir}', flush=True)
            cmd = [
                sys.executable, str(repo / 'run.py'), '--cfg', str(cfg), '--gpu', args.gpu,
                '--seed', str(args.seed), '--data', str(one_image_dir),
                '--save-root', str(save_root), '--name', name, '--bypass',
            ]
            if not args.dry_run:
                subprocess.run(cmd, cwd=repo, check=True)
            link.unlink()


if __name__ == '__main__':
    main()
