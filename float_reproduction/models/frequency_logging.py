import os
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - optional dependency
    linear_sum_assignment = None


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _as_numpy_2d(coord: torch.Tensor) -> np.ndarray:
    coord_cpu = coord.detach().cpu().float()
    if coord_cpu.ndim == 3:
        coord_cpu = coord_cpu[0]
    if coord_cpu.ndim != 2:
        raise ValueError(f"coord must be [N, D] or [B, N, D], got shape={tuple(coord_cpu.shape)}")
    return coord_cpu.numpy()


def extract_input_frequency_state(model: torch.nn.Module) -> Optional[Dict[str, np.ndarray]]:
    """Return first-layer frequency state for SIREN-like encoder, if available."""
    m = _unwrap_model(model)
    in_layer = getattr(m, "in_layer", None)
    if in_layer is None:
        return None
    linear = getattr(in_layer, "linear", None)
    if linear is None or not hasattr(linear, "weight"):
        return None
    if not hasattr(in_layer, "omega_0"):
        return None

    weight = linear.weight.detach().cpu().float().numpy()
    omega_in = float(in_layer.omega_0)
    omega = omega_in * weight
    if linear.bias is None:
        bias = np.zeros(weight.shape[0], dtype=np.float32)
    else:
        bias = linear.bias.detach().cpu().float().numpy()

    return {
        "omega_in": np.array(omega_in, dtype=np.float32),
        "weight": weight,
        "omega": omega,
        "bias": bias,
    }


def summarize_frequency_state(freq_state: Dict[str, np.ndarray]) -> Dict[str, Any]:
    weight = freq_state["weight"]
    omega = freq_state["omega"]

    weight_norm = np.linalg.norm(weight, axis=1)
    omega_norm = np.linalg.norm(omega, axis=1)
    omega_cycles = omega_norm / (2.0 * np.pi)

    summary = {
        "omega_in": float(freq_state["omega_in"]),
        "num_channels": int(weight.shape[0]),
        "input_dim": int(weight.shape[1]),
        "weight_norm": weight_norm,
        "omega_norm": omega_norm,
        "omega_cycles": omega_cycles,
        "weight_norm_mean": float(np.mean(weight_norm)),
        "weight_norm_median": float(np.median(weight_norm)),
        "weight_norm_p90": float(np.percentile(weight_norm, 90)),
        "omega_norm_mean": float(np.mean(omega_norm)),
        "omega_norm_median": float(np.median(omega_norm)),
        "omega_norm_p90": float(np.percentile(omega_norm, 90)),
        "omega_norm_max": float(np.max(omega_norm)),
        "omega_cycles_mean": float(np.mean(omega_cycles)),
        "omega_cycles_median": float(np.median(omega_cycles)),
        "omega_cycles_p90": float(np.percentile(omega_cycles, 90)),
        "omega_cycles_max": float(np.max(omega_cycles)),
    }

    if omega.shape[1] == 1:
        scalar_cycles = np.abs(omega[:, 0]) / (2.0 * np.pi)
        summary["omega_scalar_cycles"] = scalar_cycles
        summary["omega_scalar_cycles_mean"] = float(np.mean(scalar_cycles))
        summary["omega_scalar_cycles_median"] = float(np.median(scalar_cycles))
    return summary


def compute_target_frequency_alignment(
    omega_cycles: np.ndarray,
    target_frequencies: np.ndarray,
    hit_delta: float = 0.25,
) -> Dict[str, float]:
    if target_frequencies.size == 0 or omega_cycles.size == 0:
        return {}
    cycles = omega_cycles.reshape(-1)
    targets = np.asarray(target_frequencies, dtype=np.float64).reshape(-1)

    cost = np.abs(cycles[:, None] - targets[None, :])
    nearest_dist = cost.min(axis=1)
    nearest_target = cost.argmin(axis=1)
    target_nearest_dist = cost.min(axis=0)

    out = {
        "freq_align/hit_rate": float(np.mean(nearest_dist < hit_delta)),
        "freq_align/nearest_dist_mean": float(np.mean(nearest_dist)),
        "freq_align/nearest_dist_median": float(np.median(nearest_dist)),
        "freq_align/target_coverage_rate": float(np.mean(target_nearest_dist < hit_delta)),
        "freq_align/target_nearest_dist_mean": float(np.mean(target_nearest_dist)),
        "freq_align/num_targets": float(targets.size),
        "freq_align/hit_delta": float(hit_delta),
        "freq_align/nearest_target_index_mean": float(np.mean(nearest_target)),
    }

    if linear_sum_assignment is not None:
        row_idx, col_idx = linear_sum_assignment(cost)
        assign_cost = cost[row_idx, col_idx]
        out["freq_align/hungarian_assignment_cost_mean"] = float(np.mean(assign_cost))
        out["freq_align/hungarian_assignment_cost_sum"] = float(np.sum(assign_cost))
        out["freq_align/hungarian_num_matches"] = float(assign_cost.size)
    else:
        out["freq_align/hungarian_assignment_cost_mean"] = float("nan")
        out["freq_align/hungarian_assignment_cost_sum"] = float("nan")
        out["freq_align/hungarian_num_matches"] = 0.0

    return out


def _infer_grid_from_coord(coord: np.ndarray):
    if coord.shape[1] < 2:
        return None
    x = coord[:, 0]
    y = coord[:, 1]
    x_unique = np.unique(np.round(x, decimals=10))
    y_unique = np.unique(np.round(y, decimals=10))
    h = int(x_unique.size)
    w = int(y_unique.size)
    if h <= 1 or w <= 1 or h * w != coord.shape[0]:
        return None

    x_idx = np.searchsorted(x_unique, x)
    y_idx = np.searchsorted(y_unique, y)
    dx = float(np.median(np.diff(x_unique)))
    dy = float(np.median(np.diff(y_unique)))
    if dx == 0.0 or dy == 0.0:
        return None
    return h, w, x_idx, y_idx, dx, dy


def compute_line_spectrum_concentration(
    freq_state: Dict[str, np.ndarray],
    coord: torch.Tensor,
    max_channels: int = 16,
) -> Dict[str, Any]:
    """Compute how much per-channel FFT energy concentrates at expected +/-Omega bins."""
    coord_np = _as_numpy_2d(coord)
    grid_info = _infer_grid_from_coord(coord_np)
    if grid_info is None:
        return {}

    h, w, x_idx, y_idx, dx, dy = grid_info
    omega = freq_state["omega"]
    bias = freq_state["bias"]
    channels = int(omega.shape[0])
    if channels == 0:
        return {}

    num = min(max_channels, channels)
    chan_ids = np.linspace(0, channels - 1, num=num, dtype=np.int64)
    fx_bins = np.fft.fftfreq(h, d=dx)
    fy_bins = np.fft.fftfreq(w, d=dy)

    ratios = []
    peak_bins = []
    phase = coord_np @ omega[chan_ids].T + bias[chan_ids][None, :]

    for local_i, chan_i in enumerate(chan_ids):
        act_flat = np.sin(phase[:, local_i]).astype(np.float64)
        act_map = np.zeros((h, w), dtype=np.float64)
        act_map[x_idx, y_idx] = act_flat

        power = np.abs(np.fft.fft2(act_map)) ** 2
        total_power = float(np.sum(power)) + 1e-12

        if omega.shape[1] == 1:
            freq_x = float(omega[chan_i, 0] / (2.0 * np.pi))
            freq_y = 0.0
        else:
            freq_x = float(omega[chan_i, 0] / (2.0 * np.pi))
            freq_y = float(omega[chan_i, 1] / (2.0 * np.pi))

        pos_x = int(np.argmin(np.abs(fx_bins - freq_x)))
        pos_y = int(np.argmin(np.abs(fy_bins - freq_y)))
        neg_x = int(np.argmin(np.abs(fx_bins + freq_x)))
        neg_y = int(np.argmin(np.abs(fy_bins + freq_y)))

        peak_energy = float(power[pos_x, pos_y] + power[neg_x, neg_y])
        ratios.append(peak_energy / total_power)
        peak_bins.append([int(pos_x), int(pos_y), int(neg_x), int(neg_y)])

    ratios_np = np.asarray(ratios, dtype=np.float64)
    return {
        "line_spectrum/channel_ids": chan_ids,
        "line_spectrum/concentration_ratios": ratios_np,
        "line_spectrum/concentration_mean": float(np.mean(ratios_np)),
        "line_spectrum/concentration_median": float(np.median(ratios_np)),
        "line_spectrum/concentration_min": float(np.min(ratios_np)),
        "line_spectrum/concentration_max": float(np.max(ratios_np)),
        "line_spectrum/num_channels": int(num),
        "line_spectrum/h": int(h),
        "line_spectrum/w": int(w),
        "line_spectrum/peak_bins": np.asarray(peak_bins, dtype=np.int64),
    }


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (np.sum(p) + eps)
    q = q / (np.sum(q) + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return float(0.5 * (kl_pm + kl_qm))


def compute_residual_frequency_alignment(
    pred_img: torch.Tensor,
    gt_img: torch.Tensor,
    coord: torch.Tensor,
    omega_cycles: np.ndarray,
    radial_bins: int = 64,
) -> Dict[str, Any]:
    """Align residual FFT radial spectrum with learned omega radial histogram."""
    coord_np = _as_numpy_2d(coord)
    grid_info = _infer_grid_from_coord(coord_np)
    if grid_info is None:
        return {}

    h, w, _, _, dx, dy = grid_info
    pred_np = pred_img.detach().cpu().float().numpy()
    gt_np = gt_img.detach().cpu().float().numpy()

    if pred_np.ndim == 3:
        residual = (gt_np - pred_np).mean(axis=0)
    else:
        residual = gt_np - pred_np
    if residual.shape != (h, w):
        return {}

    power = np.abs(np.fft.fft2(residual.astype(np.float64))) ** 2
    fx = np.fft.fftfreq(h, d=dx)
    fy = np.fft.fftfreq(w, d=dy)
    fx_grid, fy_grid = np.meshgrid(fx, fy, indexing="ij")
    radius = np.sqrt(fx_grid ** 2 + fy_grid ** 2)

    r_flat = radius.reshape(-1)
    p_flat = power.reshape(-1)
    bins = np.linspace(0.0, float(np.max(r_flat)) + 1e-12, num=max(8, int(radial_bins)) + 1)
    residual_hist, _ = np.histogram(r_flat, bins=bins, weights=p_flat)
    omega_hist, _ = np.histogram(omega_cycles.reshape(-1), bins=bins)

    residual_hist = residual_hist.astype(np.float64)
    omega_hist = omega_hist.astype(np.float64)
    residual_norm = residual_hist / (np.sum(residual_hist) + 1e-12)
    omega_norm = omega_hist / (np.sum(omega_hist) + 1e-12)

    if np.std(residual_hist) > 0 and np.std(omega_hist) > 0:
        corr = float(np.corrcoef(residual_hist, omega_hist)[0, 1])
    else:
        corr = float("nan")

    return {
        "residual_spectrum/corr_omega_hist": corr,
        "residual_spectrum/js_divergence_omega_hist": _js_divergence(residual_norm, omega_norm),
        "residual_spectrum/radial_power_mean": float(np.mean(residual_hist)),
        "residual_spectrum/radial_power_max": float(np.max(residual_hist)),
        "residual_spectrum/radial_bins": float(len(residual_hist)),
        "residual_spectrum/h": float(h),
        "residual_spectrum/w": float(w),
        "residual_spectrum/radial_hist": residual_norm,
        "residual_spectrum/omega_hist": omega_norm,
        "residual_spectrum/radial_bin_edges": bins,
    }


def save_frequency_snapshot(
    out_dir: str,
    epoch: int,
    freq_state: Dict[str, np.ndarray],
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"omega_epoch_{int(epoch):08d}.npz")

    payload: Dict[str, Any] = {
        "epoch": np.array(int(epoch), dtype=np.int64),
        "omega_in": np.array(float(freq_state["omega_in"]), dtype=np.float32),
        "weight": freq_state["weight"].astype(np.float32),
        "omega": freq_state["omega"].astype(np.float32),
        "bias": freq_state["bias"].astype(np.float32),
    }

    if extra is not None:
        for key, value in extra.items():
            if isinstance(value, np.ndarray):
                payload[key] = value
            elif np.isscalar(value):
                payload[key] = np.array(value)

    np.savez_compressed(out_path, **payload)
    return out_path
