import torch


def matrix_info(weight, activation_lipschitz=1.0, eps=1e-12):
    """Compute matrix spectral statistics used for wandb logging."""
    with torch.no_grad():
        sv = torch.linalg.svdvals(weight.detach()) + eps
        spectral_norm = torch.max(sv)
        fro_norm_squared = torch.sum(sv ** 2)
        fro_norm = torch.sqrt(fro_norm_squared)
        stable_rank = fro_norm_squared / (spectral_norm ** 2)
        p_i = (sv ** 2) / fro_norm_squared
        shannon_entropy = -torch.sum(p_i * torch.log(p_i))
        effective_rank = torch.exp(shannon_entropy)
        spectral_condition_no = spectral_norm / torch.min(sv)

    return {
        "linear_spectral_norm": spectral_norm.item(),
        "activation_spectral_norm": float(activation_lipschitz),
        "combined_spectral_norm": spectral_norm.item() * float(activation_lipschitz),
        "frobenius_norm": fro_norm.item(),
        "stable_rank": stable_rank.item(),
        "effective_rank": effective_rank.item(),
        "spectral_condition_no": spectral_condition_no.item(),
    }


def _prepare_jacobian_input(x, max_samples=None):
    if x.ndim < 2:
        raise ValueError(f"x must have shape (..., input_dim). Got shape={tuple(x.shape)}")

    x_flat = x.reshape(-1, x.shape[-1])
    if x_flat.shape[0] == 0:
        raise ValueError("x must contain at least one sample.")

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive. Got {max_samples}")
        if x_flat.shape[0] > max_samples:
            x_flat = x_flat[:max_samples]

    return x_flat


def _compute_finite_stats(values):
    values_detached = values.detach()
    finite_vals = values_detached[torch.isfinite(values_detached)]
    if finite_vals.numel() == 0:
        return {
            "mean": float("nan"),
            "max": float("nan"),
            "min": float("nan"),
            "std": float("nan"),
        }

    return {
        "mean": finite_vals.mean().item(),
        "max": finite_vals.max().item(),
        "min": finite_vals.min().item(),
        "std": finite_vals.std(unbiased=False).item(),
    }


def jacobian_spectral_info(model, x, max_samples=None, return_per_sample=False, eps=1e-12):
    """Compute local Jacobian spectral norms ||J_f(x_i)||_2 over sampled inputs."""
    x_flat = _prepare_jacobian_input(x, max_samples=max_samples)
    x_eval = x_flat.detach().clone().requires_grad_(True)

    with torch.enable_grad():
        y = model(x_eval)
        if torch.is_complex(y):
            y = y.real
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        y_flat = y.reshape(y.shape[0], -1)

        if y_flat.shape[0] != x_eval.shape[0]:
            raise ValueError(
                "Model output batch dimension does not match input batch dimension. "
                f"input_batch={x_eval.shape[0]}, output_batch={y_flat.shape[0]}"
            )

        jacobian_rows = []
        output_dim = y_flat.shape[1]
        for out_idx in range(output_dim):
            grad = torch.autograd.grad(
                outputs=y_flat[:, out_idx].sum(),
                inputs=x_eval,
                retain_graph=(out_idx < output_dim - 1),
                create_graph=False,
            )[0]
            jacobian_rows.append(grad)

        jacobian = torch.stack(jacobian_rows, dim=1)
        sv = torch.linalg.svdvals(jacobian)
        sigma_1 = sv[..., 0]
        sigma_min = sv[..., -1]
        if sv.shape[-1] >= 2:
            sigma_2 = sv[..., 1]
        else:
            sigma_2 = torch.full_like(sigma_1, float("nan"))

    sigma_1_stats = _compute_finite_stats(sigma_1)
    sigma_2_stats = _compute_finite_stats(sigma_2)
    sigma_min_stats = _compute_finite_stats(sigma_min)

    result = {
        "jacobian_spectral_norm_mean": sigma_1_stats["mean"],
        "jacobian_spectral_norm_max": sigma_1_stats["max"],
        "jacobian_spectral_norm_min": sigma_1_stats["min"],
        "jacobian_spectral_norm_std": sigma_1_stats["std"],
        "jacobian_sigma1": sigma_1_stats["mean"],
        "jacobian_sigma2": sigma_2_stats["mean"],
        "jacobian_sigma_min": sigma_min_stats["mean"],
        "jacobian_num_samples": int(sigma_1.numel()),
        "jacobian_input_dim": int(x_eval.shape[1]),
        "jacobian_output_dim": int(y_flat.shape[1]),
    }

    if hasattr(model, "get_end_to_end_spectral_bound"):
        bound = float(model.get_end_to_end_spectral_bound())
        result["end_to_end_spectral_bound"] = bound
        result["bound_over_jacobian_max"] = bound / (result["jacobian_spectral_norm_max"] + eps)

    if return_per_sample:
        result["jacobian_spectral_norm_per_sample"] = sigma_1.detach().cpu().tolist()
        result["jacobian_sigma2_per_sample"] = sigma_2.detach().cpu().tolist()
        result["jacobian_sigma_min_per_sample"] = sigma_min.detach().cpu().tolist()

    return result


class SpectralAnalysisMixin:
    """Mixin that exposes model-level and layer-local Jacobian spectral metrics."""

    def get_jacobian_spectral_info(self, x, max_samples=None, return_per_sample=False):
        return jacobian_spectral_info(
            model=self,
            x=x,
            max_samples=max_samples,
            return_per_sample=return_per_sample,
        )

    def get_detailed_matrix_info_with_jacobian(self, x, max_samples=None, return_per_sample=False):
        info = dict(self.get_detailed_matrix_info())
        info["jacobian_spectral_info"] = self.get_jacobian_spectral_info(
            x=x,
            max_samples=max_samples,
            return_per_sample=return_per_sample,
        )
        info["layer_local_jacobian_spectral_info"] = self.get_layer_local_jacobian_spectral_info(
            x=x,
            max_samples=max_samples,
            return_per_sample=return_per_sample,
        )
        return info

    def _get_jacobian_layers(self):
        return []

    def get_layer_local_jacobian_spectral_info(self, x, max_samples=None, return_per_sample=False):
        layers = self._get_jacobian_layers()
        if not layers:
            return {"layer_local_jacobian_infos": []}

        x_eval = _prepare_jacobian_input(x, max_samples=max_samples)
        captured_inputs = {}

        def _capture_pre_input(module, inputs):
            if not inputs:
                return
            layer_input = inputs[0]
            if torch.is_tensor(layer_input):
                captured_inputs[id(module)] = layer_input.detach()

        hooks = [layer.register_forward_pre_hook(_capture_pre_input) for layer in layers]
        try:
            with torch.no_grad():
                _ = self(x_eval)
        finally:
            for hook in hooks:
                hook.remove()

        layer_infos = []
        for layer_idx, layer in enumerate(layers):
            layer_input = captured_inputs.get(id(layer), None)
            layer_type = getattr(layer, "layer_type", layer.__class__.__name__)
            layer_in_dim = int(getattr(layer, "in_features", 0))
            layer_out_dim = int(getattr(layer, "out_features", 0))

            if layer_input is None:
                layer_infos.append(
                    {
                        "layer_index": layer_idx,
                        "layer_type": layer_type,
                        "jacobian_spectral_norm_mean": float("nan"),
                        "jacobian_spectral_norm_max": float("nan"),
                        "jacobian_spectral_norm_min": float("nan"),
                        "jacobian_spectral_norm_std": float("nan"),
                        "jacobian_sigma1": float("nan"),
                        "jacobian_sigma2": float("nan"),
                        "jacobian_sigma_min": float("nan"),
                        "jacobian_num_samples": 0,
                        "jacobian_input_dim": layer_in_dim,
                        "jacobian_output_dim": layer_out_dim,
                    }
                )
                continue

            try:
                local_info = layer.get_local_jacobian_spectral_info(
                    x=layer_input,
                    max_samples=None,
                    return_per_sample=return_per_sample,
                )
            except Exception:
                local_info = {
                    "jacobian_spectral_norm_mean": float("nan"),
                    "jacobian_spectral_norm_max": float("nan"),
                    "jacobian_spectral_norm_min": float("nan"),
                    "jacobian_spectral_norm_std": float("nan"),
                    "jacobian_sigma1": float("nan"),
                    "jacobian_sigma2": float("nan"),
                    "jacobian_sigma_min": float("nan"),
                    "jacobian_num_samples": int(layer_input.shape[0]),
                    "jacobian_input_dim": int(layer_input.shape[-1]),
                    "jacobian_output_dim": layer_out_dim,
                }

            local_info["layer_index"] = layer_idx
            local_info["layer_type"] = layer_type
            layer_infos.append(local_info)

        return {
            "layer_local_jacobian_infos": layer_infos,
            "num_layers": len(layer_infos),
        }
