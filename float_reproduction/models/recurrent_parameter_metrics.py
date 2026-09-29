import math

import torch
import torch.nn.functional as F

from models.spectral_metrics import matrix_info


def _extract_recurrent_module(model):
    if hasattr(model, "recurrnet_net"):
        return model.recurrnet_net
    return None


def _extract_weight(module, mode="full"):
    if hasattr(module, "_get_weight"):
        return module._get_weight(mode=mode)
    if mode != "full":
        return None
    if hasattr(module, "linear") and hasattr(module.linear, "weight"):
        return module.linear.weight
    if hasattr(module, "weight"):
        return module.weight
    return None


def _available_modes(module):
    full_weight = _extract_weight(module, mode="full")
    if full_weight is None:
        return []

    modes = ["full"]
    if full_weight.ndim == 2 and full_weight.shape[0] == full_weight.shape[1]:
        sym_weight = _extract_weight(module, mode="symmetric")
        asym_weight = _extract_weight(module, mode="antisymmetric")
        if sym_weight is not None and asym_weight is not None:
            modes.extend(["symmetric", "antisymmetric"])
    return modes


def _sanitize_schedule(schedule, fallback_mode, available_modes):
    if not schedule:
        return [fallback_mode]
    sanitized = []
    for mode in schedule:
        if mode in available_modes:
            sanitized.append(mode)
        else:
            sanitized.append(fallback_mode)
    return sanitized


def _get_recurrent_schedule(model, available_modes):
    fallback_mode = "full" if "full" in available_modes else (available_modes[0] if available_modes else "full")
    if hasattr(model, "get_recurrent_analysis_schedule"):
        try:
            raw_schedule = model.get_recurrent_analysis_schedule()
        except Exception:
            raw_schedule = None
        return _sanitize_schedule(raw_schedule, fallback_mode, available_modes)

    recurrent_steps = int(getattr(model, "recurrent_steps", 1))
    return [fallback_mode for _ in range(max(1, recurrent_steps))]


def _pairwise_weight_change_stats(current_w, prev_w, init_w, eps=1e-12):
    current_vec = current_w.detach().reshape(-1).float()
    prev_vec = prev_w.detach().reshape(-1).float()
    init_vec = init_w.detach().reshape(-1).float()

    delta_prev = current_vec - prev_vec
    delta_init = current_vec - init_vec

    prev_norm = torch.norm(prev_vec, p=2)
    init_norm = torch.norm(init_vec, p=2)
    delta_prev_norm = torch.norm(delta_prev, p=2)
    delta_init_norm = torch.norm(delta_init, p=2)

    cosine_prev = F.cosine_similarity(
        current_vec.unsqueeze(0),
        prev_vec.unsqueeze(0),
        dim=1,
        eps=eps,
    ).item()
    cosine_init = F.cosine_similarity(
        current_vec.unsqueeze(0),
        init_vec.unsqueeze(0),
        dim=1,
        eps=eps,
    ).item()

    return {
        "update_l2_from_prev": float(delta_prev_norm.item()),
        "relative_update_from_prev": float((delta_prev_norm / (prev_norm + eps)).item()),
        "drift_l2_from_init": float(delta_init_norm.item()),
        "relative_drift_from_init": float((delta_init_norm / (init_norm + eps)).item()),
        "cosine_to_prev": float(cosine_prev),
        "cosine_to_init": float(cosine_init),
    }


def recurrent_parameter_dynamics_info(model, eps=1e-12):
    recurrent_module = _extract_recurrent_module(model)
    if recurrent_module is None:
        return {}

    modes = _available_modes(recurrent_module)
    if not modes:
        return {}

    omega = abs(float(getattr(recurrent_module, "omega_0", 1.0)))
    schedule = _get_recurrent_schedule(model, available_modes=modes)
    tracked_modes = list(dict.fromkeys(modes + schedule))

    state = getattr(model, "_recurrent_param_analysis_state", None)
    if state is None:
        state = {"prev": {}, "init": {}}

    mode_infos = {}
    flat_metrics = {}
    mode_lipschitz = {}
    schedule_counts = {}
    for mode in schedule:
        schedule_counts[mode] = schedule_counts.get(mode, 0) + 1

    for mode in tracked_modes:
        weight = _extract_weight(recurrent_module, mode=mode)
        if weight is None:
            continue

        weight_cpu = weight.detach().cpu().clone()
        prev_w = state["prev"].get(mode, None)
        init_w = state["init"].get(mode, None)
        if init_w is None:
            init_w = weight_cpu.clone()
            state["init"][mode] = init_w
        if prev_w is None:
            prev_w = weight_cpu.clone()

        info = matrix_info(weight, activation_lipschitz=omega)
        step_lipschitz = float(info["combined_spectral_norm"])
        contraction_margin = float(1.0 - step_lipschitz)
        is_contractive = float(step_lipschitz < 1.0)
        step_log_bound = float(math.log(max(step_lipschitz, eps)))

        info.update(
            {
                "per_step_lipschitz_bound": step_lipschitz,
                "contraction_margin": contraction_margin,
                "is_contractive": is_contractive,
                "log_step_lipschitz_bound": step_log_bound,
                "omega_0": float(omega),
            }
        )
        info.update(_pairwise_weight_change_stats(weight_cpu, prev_w, init_w, eps=eps))

        use_count = int(schedule_counts.get(mode, 0))
        if use_count > 0:
            info["scheduled_step_count"] = use_count
            info["scheduled_compound_bound"] = float(step_lipschitz ** use_count)
            info["scheduled_compound_log_bound"] = float(use_count * step_log_bound)
        else:
            info["scheduled_step_count"] = 0
            info["scheduled_compound_bound"] = 1.0
            info["scheduled_compound_log_bound"] = 0.0

        mode_infos[mode] = info
        mode_lipschitz[mode] = step_lipschitz
        state["prev"][mode] = weight_cpu

        for key, value in info.items():
            if isinstance(value, (int, float)):
                flat_metrics[f"recurrent_param/{mode}/{key}"] = float(value)

    schedule_bound = 1.0
    schedule_log_bound = 0.0
    step_bounds = []
    for mode in schedule:
        step_bound = float(mode_lipschitz.get(mode, 1.0))
        step_bounds.append(step_bound)
        schedule_bound *= step_bound
        schedule_log_bound += math.log(max(step_bound, eps))

    if step_bounds:
        flat_metrics["recurrent_param/schedule/num_steps"] = float(len(step_bounds))
        flat_metrics["recurrent_param/schedule/max_step_lipschitz"] = float(max(step_bounds))
        flat_metrics["recurrent_param/schedule/min_step_lipschitz"] = float(min(step_bounds))
        flat_metrics["recurrent_param/schedule/mean_step_lipschitz"] = float(sum(step_bounds) / len(step_bounds))
        flat_metrics["recurrent_param/schedule/compound_lipschitz_bound"] = float(schedule_bound)
        flat_metrics["recurrent_param/schedule/log_compound_lipschitz_bound"] = float(schedule_log_bound)
        flat_metrics["recurrent_param/schedule/is_fully_contractive"] = float(max(step_bounds) < 1.0)

    if "full" in mode_infos and "symmetric" in mode_infos:
        full_frob = mode_infos["full"]["frobenius_norm"]
        sym_frob = mode_infos["symmetric"]["frobenius_norm"]
        asym_frob = mode_infos.get("antisymmetric", {}).get("frobenius_norm", 0.0)
        full_energy = max(full_frob ** 2, eps)
        flat_metrics["recurrent_param/decomposition/symmetric_energy_ratio"] = float((sym_frob ** 2) / full_energy)
        flat_metrics["recurrent_param/decomposition/antisymmetric_energy_ratio"] = float((asym_frob ** 2) / full_energy)
        flat_metrics["recurrent_param/decomposition/symmetry_gap_l2"] = float(
            abs(mode_infos["full"]["linear_spectral_norm"] - mode_infos["symmetric"]["linear_spectral_norm"])
        )

    setattr(model, "_recurrent_param_analysis_state", state)
    return flat_metrics
