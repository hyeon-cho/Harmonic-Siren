import torch
import torch.nn as nn

import numpy as np


class SineLayer(nn.Module):
    def __init__(self, 
                 in_features  : int, 
                 out_features : int, 
                 bias         : bool=True,
                 is_first     : bool=False, 
                 omega_0      : int=30):
        super().__init__()
        self.omega_0     = omega_0
        self.is_first    = is_first
        self.in_features = in_features

        self.linear      = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x)) 
