"""Paper-style plots for the SDF showcase matrix.

Inputs: paper_results/curves_gam.csv and curves_sine.csv.
Outputs: plots/fig_chamfer_vs_epoch, fig_components, and SUMMARY.md.
Wall-clock and train-loss figures are generated when their original logs exist.
"""
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'paper_results')
OUT = os.path.join(ROOT, 'plots')
os.makedirs(OUT, exist_ok=True)

METHODS = {
    'mat_siren':   dict(label='SIREN (lr 1e-4)',            color='#888888', ls='--'),
    'mat_gam1e-4': dict(label='Ours, R=3 (lr 1e-4, paper)', color='#1f77b4', ls='-'),
    'mat_gam5e-4': dict(label='Ours, R=3 (lr 5e-4)',        color='#d62728', ls='-'),
}
SEEDS = [0, 1, 2]

plt.rcParams.update({
    'figure.dpi': 120, 'savefig.dpi': 300, 'font.size': 11,
    'axes.labelsize': 11, 'legend.fontsize': 9.5, 'axes.linewidth': 1.0,
    'lines.linewidth': 2.0,
})


def clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which='major', alpha=0.25)
    ax.grid(True, which='minor', alpha=0.10)


def method_of(run):
    return re.sub(r'_s\d+$', '', run)


def load_curves():
    data = defaultdict(dict)  # method -> seed -> {epoch: (cd, comp)}
    for path in glob.glob(os.path.join(DATA, 'curves_*.csv')):
        for row in csv.DictReader(open(path)):
            run = row['run']
            m = method_of(run)
            s = int(run.rsplit('_s', 1)[1])
            ep = int(row['epoch'])
            if ep < 0:
                ep = 2000
            cd = float(row['chamfer_x1e6'])
            comp = int(row['components'])
            data[m].setdefault(s, {})[ep] = (cd, comp)
    return data


def band(ax, xs, mat, color, label, ls='-'):
    mean = np.nanmean(mat, axis=0)
    lo, hi = np.nanmin(mat, axis=0), np.nanmax(mat, axis=0)
    ax.plot(xs, mean, color=color, label=label, ls=ls)
    ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0)


def mean_iter_time(method):
    ts = []
    for s in SEEDS:
        for cand in (f'{method}_s{s}.out', f'mat_{method}_s{s}.out'):
            f = os.path.join(ROOT, 'logs_run', cand)
            if os.path.exists(f):
                vals = re.findall(r'iteration time ([0-9.]+)', open(f, errors='ignore').read())
                ts += [float(v) for v in vals[1:]]  # drop first (warmup)
                break
    return float(np.mean(ts)) if ts else float('nan')


def main():
    data = load_curves()
    if not data:
        raise FileNotFoundError(f'no curve CSVs in {DATA}')
    epochs = sorted({e for m in data.values() for s in m.values() for e in s})

    # ---- Fig 1: Chamfer vs epoch ----
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    clean_axes(ax)
    for m, sty in METHODS.items():
        if m not in data:
            continue
        mat = np.array([[data[m].get(s, {}).get(e, (np.nan,))[0] for e in epochs] for s in SEEDS])
        band(ax, epochs, mat, sty['color'], sty['label'], sty['ls'])
    ax.set_yscale('log')
    ax.set_xlabel('Training epoch')
    ax.set_ylabel(r'Chamfer distance ($\times 10^{-6}$)')
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_chamfer_vs_epoch.{ext}'), bbox_inches='tight')
    plt.close(fig)

    # ---- Fig 2: Chamfer vs wall-clock ----
    times = {m: mean_iter_time(m) for m in METHODS}
    if any(np.isfinite(t) for t in times.values()):
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        clean_axes(ax)
        for m, sty in METHODS.items():
            if m not in data or not np.isfinite(times[m]):
                continue
            t = times[m]
            mat = np.array([[data[m].get(s, {}).get(e, (np.nan,))[0] for e in epochs] for s in SEEDS])
            band(ax, np.array(epochs) * t / 60.0, mat, sty['color'],
                 f"{sty['label']}  ({t:.2f}s/ep)", sty['ls'])
        ax.set_yscale('log')
        ax.set_xlabel('Wall-clock training time (min)')
        ax.set_ylabel(r'Chamfer distance ($\times 10^{-6}$)')
        ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(OUT, f'fig_chamfer_vs_time.{ext}'), bbox_inches='tight')
        plt.close(fig)

    # ---- Fig 3: train loss ----
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    clean_axes(ax)
    has_train_loss = False
    for m, sty in METHODS.items():
        curves = []
        for s in SEEDS:
            f = os.path.join(ROOT, 'logs', f'{m}_s{s}', 'checkpoints', 'train_losses_final.txt')
            if os.path.exists(f):
                curves.append(np.loadtxt(f))
        if not curves:
            continue
        has_train_loss = True
        L = min(map(len, curves))
        mat = np.stack([c[:L] for c in curves])
        xs = np.arange(1, L + 1)
        band(ax, xs, mat, sty['color'], sty['label'], sty['ls'])
    if has_train_loss:
        ax.set_yscale('log')
        ax.set_xlabel('Training epoch')
        ax.set_ylabel('Training loss (SDF objective)')
        ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(OUT, f'fig_train_loss.{ext}'), bbox_inches='tight')
    plt.close(fig)

    # ---- Fig 4: surface components (artifact indicator) ----
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    clean_axes(ax)
    for m, sty in METHODS.items():
        if m not in data:
            continue
        mat = np.array([[data[m].get(s, {}).get(e, (np.nan, np.nan))[1] for e in epochs] for s in SEEDS],
                       dtype=float)
        band(ax, epochs, mat, sty['color'], sty['label'], sty['ls'])
    ax.axhline(1.0, color='k', lw=0.8, alpha=0.5)
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('# disconnected surface components')
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_components.{ext}'), bbox_inches='tight')
    plt.close(fig)

    # ---- summary table ----
    marks = [100, 200, 500, 1000, 1500, 2000]
    lines = ['# SDF showcase summary (Armadillo, 3 seeds, mean ± std, CD x 1e-6)\n',
             '| method | ' + ' | '.join(f'ep{e}' for e in marks) + ' | final #comp |',
             '|---|' + '---|' * (len(marks) + 1)]
    for m, sty in METHODS.items():
        if m not in data:
            continue
        cells = []
        for e in marks:
            vals = [data[m].get(s, {}).get(e, (np.nan,))[0] for s in SEEDS]
            cells.append(f'{np.nanmean(vals):.0f} ± {np.nanstd(vals):.0f}')
        comps = [data[m].get(s, {}).get(2000, (np.nan, np.nan))[1] for s in SEEDS]
        lines.append(f"| {sty['label']} | " + ' | '.join(cells) + f' | {np.nanmean(comps):.1f} |')
    open(os.path.join(OUT, 'SUMMARY.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nplots written to {OUT}/')


if __name__ == '__main__':
    main()
