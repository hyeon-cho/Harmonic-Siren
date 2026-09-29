import torch
import torch.nn as nn

import numpy as np

from models.spectral_metrics import jacobian_spectral_info, matrix_info


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: int = 30,
        layer_type: str = None,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features
        self.layer_type = "first" if layer_type is None and is_first else (layer_type or "hidden")

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

    def get_info(self):
        activation_lipschitz = abs(float(self.omega_0))
        return matrix_info(self.linear.weight, activation_lipschitz=activation_lipschitz)

    def get_local_jacobian_spectral_info(self, x, max_samples=None, return_per_sample=False):
        return jacobian_spectral_info(
            model=self,
            x=x,
            max_samples=max_samples,
            return_per_sample=return_per_sample,
        )


class LinearWithMetrics(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, layer_type="last"):
        super().__init__(in_features, out_features, bias=bias)
        self.layer_type = layer_type

    def get_info(self):
        return matrix_info(self.weight, activation_lipschitz=1.0)

    def get_local_jacobian_spectral_info(self, x, max_samples=None, return_per_sample=False):
        return jacobian_spectral_info(
            model=self,
            x=x,
            max_samples=max_samples,
            return_per_sample=return_per_sample,
        )
