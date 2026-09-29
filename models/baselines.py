"""INR baselines used for paper comparisons.

All baselines output 3-channel RGB and use plain forward(coord); they only
make sense under the trainer's `rec` mode (no bit decomposition).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.models import register


###############################################################################
# Common: SIREN-style sine layer (re-defined locally, identical to models.modules.SineLayer)
###############################################################################
class _SineLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0     = omega_0
        self.is_first    = is_first
        self.in_channels = in_channels
        self.linear      = nn.Linear(in_channels, out_channels, bias=bias)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_channels,
                                             1 / self.in_channels)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_channels) / self.omega_0,
                                             np.sqrt(6 / self.in_channels) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


###############################################################################
# SIREN (vanilla, num_layers sequential SineLayers)
###############################################################################
@register('SIREN')
class SIREN(nn.Module):
    def __init__(self,
                 in_channels=2, hidden_channels=256, out_channels=3,
                 bias=False, num_layers=5, outer_most_linear=True,
                 omega_0={'first': 30., 'hidden': 30.}, **_unused):
        super().__init__()
        nets = [_SineLayer(in_channels, hidden_channels, bias=bias,
                           is_first=True, omega_0=omega_0['first'])]
        for _ in range(num_layers - 2):
            nets.append(_SineLayer(hidden_channels, hidden_channels, bias=bias,
                                   is_first=False, omega_0=omega_0['hidden']))
        if outer_most_linear:
            final = nn.Linear(hidden_channels, out_channels, bias=bias)
            with torch.no_grad():
                final.weight.uniform_(-np.sqrt(6 / hidden_channels) / omega_0['hidden'],
                                       np.sqrt(6 / hidden_channels) / omega_0['hidden'])
            nets.append(final)
        else:
            nets.append(_SineLayer(hidden_channels, out_channels, bias=bias,
                                   is_first=False, omega_0=omega_0['hidden']))
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


###############################################################################
# RecurrentSIREN: 1 input SineLayer + 1 weight-tied SineLayer applied num_layers times + 1 final
###############################################################################
@register('RecurrentSIREN')
class RecurrentSIREN(nn.Module):
    def __init__(self,
                 in_channels=2, hidden_channels=256, out_channels=3,
                 bias=False, num_layers=5, outer_most_linear=True,
                 omega_0={'first': 30., 'hidden': 30.}, **_unused):
        super().__init__()
        self.num_layers = num_layers
        self.input_layer = _SineLayer(in_channels, hidden_channels, bias=bias,
                                       is_first=True, omega_0=omega_0['first'])
        self.recurrent_layer = _SineLayer(hidden_channels, hidden_channels, bias=bias,
                                           is_first=False, omega_0=omega_0['hidden'])
        if outer_most_linear:
            self.final_layer = nn.Linear(hidden_channels, out_channels, bias=bias)
            with torch.no_grad():
                self.final_layer.weight.uniform_(-np.sqrt(6 / hidden_channels) / omega_0['hidden'],
                                                  np.sqrt(6 / hidden_channels) / omega_0['hidden'])
        else:
            self.final_layer = _SineLayer(hidden_channels, out_channels, bias=bias,
                                           is_first=False, omega_0=omega_0['hidden'])

    def forward(self, x):
        x = self.input_layer(x)
        for _ in range(self.num_layers):
            x = self.recurrent_layer(x)
        return self.final_layer(x)


###############################################################################
# FINER (https://arxiv.org/abs/2312.02434)
###############################################################################
class _FinerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, is_first=False,
                 omega_0=30, first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_channels = in_channels
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self._init_weights()
        self.scale_req_grad = scale_req_grad
        self.first_bias_scale = first_bias_scale
        if self.first_bias_scale is not None:
            self._init_first_bias()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_channels,
                                             1 / self.in_channels)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_channels) / self.omega_0,
                                             np.sqrt(6 / self.in_channels) / self.omega_0)

    def _init_first_bias(self):
        with torch.no_grad():
            if self.is_first and self.linear.bias is not None:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)

    def _generate_scale(self, x):
        if self.scale_req_grad:
            return torch.abs(x) + 1
        with torch.no_grad():
            return torch.abs(x) + 1

    def forward(self, x):
        x = self.linear(x)
        scale = self._generate_scale(x)
        return torch.sin(self.omega_0 * scale * x)


@register('FINER')
class FINER(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=512, hidden_layers=5,
                 out_channels=3,
                 omega_0={'first': 30., 'hidden': 30.},
                 bias=True, first_bias_scale=None, scale_req_grad=False, **_unused):
        super().__init__()
        net = [_FinerLayer(in_channels, hidden_channels, bias=bias, is_first=True,
                           omega_0=omega_0['first'],
                           first_bias_scale=first_bias_scale,
                           scale_req_grad=scale_req_grad)]
        for _ in range(hidden_layers):
            net.append(_FinerLayer(hidden_channels, hidden_channels, bias=bias,
                                   omega_0=omega_0['hidden'],
                                   scale_req_grad=scale_req_grad))
        final = nn.Linear(hidden_channels, out_channels)
        with torch.no_grad():
            final.weight.uniform_(-np.sqrt(6 / hidden_channels) / omega_0['hidden'],
                                   np.sqrt(6 / hidden_channels) / omega_0['hidden'])
        net.append(final)
        self.net = nn.Sequential(*net)

    def forward(self, coords):
        return self.net(coords)


@register('FINER_Rec')
class FINER_Rec(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=512, hidden_layers=5,
                 out_channels=3,
                 omega_0={'first': 30., 'hidden': 30.},
                 bias=True, first_bias_scale=None, scale_req_grad=False, **_unused):
        super().__init__()
        self.n_recs = hidden_layers
        self.inp_layer = _FinerLayer(in_channels, hidden_channels, bias=bias, is_first=True,
                                       omega_0=omega_0['first'],
                                       first_bias_scale=first_bias_scale,
                                       scale_req_grad=scale_req_grad)
        self.rec_layer = _FinerLayer(hidden_channels, hidden_channels, bias=bias,
                                       omega_0=omega_0['hidden'],
                                       scale_req_grad=scale_req_grad)
        self.final_linear = nn.Linear(hidden_channels, out_channels)
        with torch.no_grad():
            self.final_linear.weight.uniform_(-np.sqrt(6 / hidden_channels) / omega_0['hidden'],
                                               np.sqrt(6 / hidden_channels) / omega_0['hidden'])

    def forward(self, coords):
        x = self.inp_layer(coords)
        for _ in range(self.n_recs):
            x = self.rec_layer(x)
        return self.final_linear(x)


###############################################################################
# WIRE (https://github.com/vishwa91/wire) — complex Gabor
###############################################################################
class _ComplexGaborLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, is_first=False,
                 omega0=10.0, sigma0=40.0, trainable=False):
        super().__init__()
        self.is_first = is_first
        self.in_channels = in_channels

        dtype = torch.float if is_first else torch.cfloat
        self.omega_0 = nn.Parameter(omega0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(sigma0 * torch.ones(1), trainable)
        self.linear = nn.Linear(in_channels, out_channels, bias=bias, dtype=dtype)

    def forward(self, x):
        lin = self.linear(x)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        return torch.exp(1j * omega - scale.abs().square())


@register('WIRE')
class WIRE(nn.Module):
    def __init__(self, in_channels=2, out_channels=3, hidden_channels=256, hidden_layers=5,
                 omega_0={'first': 20., 'hidden': 20.}, scale=10.0, **_unused):
        super().__init__()
        # complex hidden -> divide channel by sqrt(2) so total real params ≈ hidden_channels^2 per layer
        h = int(hidden_channels / np.sqrt(2))
        net = [_ComplexGaborLayer(in_channels, h, is_first=True,
                                   omega0=omega_0['first'], sigma0=scale)]
        for _ in range(hidden_layers):
            net.append(_ComplexGaborLayer(h, h, omega0=omega_0['hidden'], sigma0=scale))
        net.append(nn.Linear(h, out_channels, dtype=torch.cfloat))
        self.net = nn.Sequential(*net)

    def forward(self, coords):
        return self.net(coords).real


###############################################################################
# Plain ReLU MLP and ReLU+Positional Encoding MLP (PEMLP)
###############################################################################
@register('ReLU_MLP')
class ReLU_MLP(nn.Module):
    """Vanilla ReLU MLP. num_layers = total Linear layers (1 input + (num_layers-2) hidden + 1 output)."""
    def __init__(self, in_channels=2, hidden_channels=256, out_channels=3,
                 num_layers=5, **_unused):
        super().__init__()
        net = [nn.Linear(in_channels, hidden_channels), nn.ReLU(True)]
        for _ in range(num_layers - 2):
            net.append(nn.Linear(hidden_channels, hidden_channels))
            net.append(nn.ReLU(True))
        net.append(nn.Linear(hidden_channels, out_channels))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class _PositionalEncoding(nn.Module):
    """log-scale (sin, cos) positional encoding of input coords."""
    def __init__(self, in_channels, N_freqs, logscale=True):
        super().__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)
        if logscale:
            freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(freq * x))
        return torch.cat(out, -1)


@register('PEMLP')
class PEMLP(nn.Module):
    """Positional-encoding ReLU MLP (NeRF-style PE).
    num_layers = number of hidden Linear layers (1 input from PE + num_layers hidden + 1 output)."""
    def __init__(self, in_channels=2, hidden_channels=256, out_channels=3,
                 num_layers=5, N_freqs=10, **_unused):
        super().__init__()
        self.encoding = _PositionalEncoding(in_channels=in_channels, N_freqs=N_freqs)
        net = [nn.Linear(self.encoding.out_channels, hidden_channels), nn.ReLU(True)]
        for _ in range(num_layers):
            net.append(nn.Linear(hidden_channels, hidden_channels))
            net.append(nn.ReLU(True))
        net.append(nn.Linear(hidden_channels, out_channels))
        self.net = nn.Sequential(*net)

    def forward(self, coords):
        return self.net(self.encoding(coords))


###############################################################################
# Gauss MLP (https://github.com/sergiortiz3/gauss-act, INR with exp(-(σ·Wx)²))
###############################################################################
class _GaussLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, scale=30.0):
        super().__init__()
        self.scale = scale
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        return torch.exp(-(self.scale * self.linear(x)) ** 2)


@register('Gauss')
class Gauss(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=256, out_channels=3,
                 hidden_layers=3, scale=30.0, **_unused):
        super().__init__()
        net = [_GaussLayer(in_channels, hidden_channels, scale=scale)]
        for _ in range(hidden_layers):
            net.append(_GaussLayer(hidden_channels, hidden_channels, scale=scale))
        net.append(nn.Linear(hidden_channels, out_channels))
        self.net = nn.Sequential(*net)

    def forward(self, coords):
        return self.net(coords)


###############################################################################
# Multiplicative Filter Networks (Fathony et al. ICLR 2021)
# Ported from https://github.com/boschresearch/multiplicative-filter-networks
###############################################################################
class _MFNBase(nn.Module):
    """Common MFN backbone: filters[0] -> filters[i] * linear[i-1] -> ... -> output_linear."""
    def __init__(self, hidden_size, out_size, n_layers, weight_scale, bias=True, output_act=False):
        super().__init__()
        self.linear = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias) for _ in range(n_layers)])
        self.output_linear = nn.Linear(hidden_size, out_size)
        self.output_act = output_act
        for lin in self.linear:
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / hidden_size),
                 np.sqrt(weight_scale / hidden_size),
            )

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i - 1](out)
        out = self.output_linear(out)
        if self.output_act:
            out = torch.sin(out)
        return out


class _FourierLayerMFN(nn.Module):
    """Fourier filter from FourierNet (no bandlimit)."""
    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.data *= weight_scale
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        return torch.sin(self.linear(x))


@register('FourierNet')
class FourierNet(_MFNBase):
    def __init__(self, in_channels=2, hidden_channels=256, out_channels=3,
                 n_layers=3, input_scale=256.0, weight_scale=1.0,
                 bias=True, output_act=False, **_unused):
        super().__init__(hidden_channels, out_channels, n_layers, weight_scale, bias, output_act)
        self.filters = nn.ModuleList([
            _FourierLayerMFN(in_channels, hidden_channels,
                             input_scale / np.sqrt(n_layers + 1))
            for _ in range(n_layers + 1)
        ])


class _GaborLayer(nn.Module):
    """Gabor filter from GaborNet."""
    def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,)))
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :])


@register('GaborNet')
class GaborNet(_MFNBase):
    def __init__(self, in_channels=2, hidden_channels=256, out_channels=3,
                 n_layers=3, input_scale=256.0, weight_scale=1.0,
                 alpha=6.0, beta=1.0,
                 bias=True, output_act=False, **_unused):
        super().__init__(hidden_channels, out_channels, n_layers, weight_scale, bias, output_act)
        self.filters = nn.ModuleList([
            _GaborLayer(in_channels, hidden_channels,
                        input_scale / np.sqrt(n_layers + 1),
                        alpha / (n_layers + 1), beta)
            for _ in range(n_layers + 1)
        ])


###############################################################################
# BACON (Lindell et al. CVPR 2022) — Band-limited Coordinate Networks
# Ported (single-scale variant) from https://github.com/computational-imaging/bacon
###############################################################################
import math as _math
class _FourierLayerBACON(nn.Module):
    """FourierLayer with quantized (bandlimited) frequency init from BACON.

    weight_scale is per-input-dim (list/tuple of length in_features). Frequencies
    are sampled from a discrete uniform on multiples of `quantization_interval`
    in [-weight_scale[d], +weight_scale[d]] per input dim d. Weights are frozen
    (requires_grad=False); only the bias is trainable.
    """
    def __init__(self, in_features, out_features, weight_scale, quantization_interval=2 * np.pi):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if not hasattr(weight_scale, '__len__'):
            weight_scale = [weight_scale] * in_features
        for i in range(in_features):
            r = 2 * weight_scale[i] / quantization_interval
            assert _math.isclose(r, round(r)), \
                f'weight_scale[{i}]={weight_scale[i]} not divisible by quantization_interval={quantization_interval}'
            init = torch.randint_like(self.linear.weight.data[:, i],
                                      0, int(2 * weight_scale[i] / quantization_interval) + 1)
            init = init * quantization_interval - weight_scale[i]
            self.linear.weight.data[:, i] = init
        self.linear.weight.requires_grad = False
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        return torch.sin(self.linear(x))


@register('BACON')
class BACON(_MFNBase):
    """Single-scale BACON: bandlimited multiplicative filter network.

    Frequencies for each layer are bounded by frequency/(n_layers+1) so the
    composed network bandwidth is guaranteed to be <= `frequency`.
    """
    def __init__(self, in_channels=2, hidden_channels=256, out_channels=3,
                 hidden_layers=3, weight_scale=1.0, bias=True, output_act=False,
                 frequency=(128.0, 128.0),
                 quantization_interval=2 * np.pi,
                 **_unused):
        super().__init__(hidden_channels, out_channels, hidden_layers,
                         weight_scale, bias, output_act)
        # input_scale per input dim, snapped to a multiple of quantization_interval
        input_scale = [round((np.pi * f / (hidden_layers + 1)) / quantization_interval)
                       * quantization_interval for f in frequency]
        self.filters = nn.ModuleList([
            _FourierLayerBACON(in_channels, hidden_channels, input_scale,
                               quantization_interval=quantization_interval)
            for _ in range(hidden_layers + 1)
        ])


###############################################################################
# (Implicit)^2 — DEQ-backed MFN INR. Huang, Bai, Kolter NeurIPS 2021.
# Ported (minimal: forward-iteration solver + 1-step backprop) from
#   https://github.com/locuslab/ImpSq
###############################################################################
class _ImpSqFourierFilter(nn.Module):
    """sin-filter used as the multiplicative gate: scale = sin(W x + b), out = scale * z."""
    def __init__(self, in_features, interm_features, scale=1.0):
        super().__init__()
        self.x_map = nn.Linear(in_features, interm_features)
        self.x_map.weight.data *= scale
        self.x_map.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x, z):
        return torch.sin(self.x_map(x)) * z


class _ImpSqMFNLayer(nn.Module):
    """One layer of the DEQ inner-cell: z' = filter(x) * (W z + b)."""
    def __init__(self, in_features, interm_features, scale_per_layer):
        super().__init__()
        self.z_map = nn.Linear(interm_features, interm_features)
        nn.init.uniform_(self.z_map.weight,
                         -np.sqrt(1.0 / interm_features),
                          np.sqrt(1.0 / interm_features))
        self.filter = _ImpSqFourierFilter(in_features, interm_features, scale_per_layer)

    def forward(self, x, z):
        return self.filter(x, self.z_map(z))


class _ImpSqMFN(nn.Module):
    """Inner cell f(x, z) for the DEQ:  z_new = init_filter(x) + L_n(L_{n-1}(... L_1(z))).
    On the first call the init_filter is added (only depends on x, equivalent to a bias)."""
    def __init__(self, in_features, interm_features, n_layers, input_scale=256.0):
        super().__init__()
        layer_scale = float(np.sqrt(n_layers + 1))
        self.init_filter = _ImpSqFourierFilter(in_features, interm_features,
                                                input_scale / layer_scale)
        self.layers = nn.ModuleList([
            _ImpSqMFNLayer(in_features, interm_features, input_scale / layer_scale)
            for _ in range(n_layers)
        ])
        self.register_buffer('init_filter_in', torch.ones(1, interm_features))
        self.interm_features = interm_features

    def forward(self, x, z):
        # init_filter contributes a fixed (per-x) bias to the fixed-point equation
        gx = self.init_filter(x, self.init_filter_in)
        z = z + gx
        for layer in self.layers:
            z = layer(x, z)
        return z


class _ImpSqGaborFilter(nn.Module):
    """Gabor inner filter for iGabor: scale = sin(W x + b) * exp(-0.5 * gamma * ||x-mu||^2), out = scale * z."""
    def __init__(self, in_features, interm_features, scale=1.0, alpha=5.0, beta=1.0):
        super().__init__()
        self.x_map = nn.Linear(in_features, interm_features)
        self.mu = nn.Parameter(torch.rand(interm_features, in_features) * 2 - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((interm_features,)))
        self.x_map.weight.data *= scale * torch.sqrt(self.gamma[:, None])
        self.x_map.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x, z):
        # ||x - mu||^2 with x: (..., in), mu: (interm, in)
        D = (x ** 2).sum(-1, keepdim=True) + (self.mu ** 2).sum(-1)[None, :] - 2 * x @ self.mu.T
        scale = torch.sin(self.x_map(x)) * torch.exp(-0.5 * F.relu(self.gamma)[None, :] * D)
        return scale * z


class _ImpSqSirenFilter(nn.Module):
    """SIREN-like inner filter for iSIREN: out = sin(z + W x + b)."""
    def __init__(self, in_features, interm_features, scale=1.0):
        super().__init__()
        self.x_map = nn.Linear(in_features, interm_features)
        self.x_map.weight.data *= scale
        self.x_map.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x, z):
        return torch.sin(z + self.x_map(x))


def _make_impsq_filter(filter_type, in_features, interm_features, scale, **opts):
    if filter_type == 'fourier':    return _ImpSqFourierFilter(in_features, interm_features, scale)
    if filter_type == 'gabor':      return _ImpSqGaborFilter(in_features, interm_features, scale,
                                                              alpha=opts.get('alpha', 5.0),
                                                              beta=opts.get('beta', 1.0))
    if filter_type == 'siren_like': return _ImpSqSirenFilter(in_features, interm_features, scale)
    raise ValueError(filter_type)


class _ImpSqMFNLayerGeneric(nn.Module):
    """One layer of the DEQ inner cell, with arbitrary inner filter type."""
    def __init__(self, in_features, interm_features, scale_per_layer, filter_type='fourier', **opts):
        super().__init__()
        self.z_map = nn.Linear(interm_features, interm_features)
        nn.init.uniform_(self.z_map.weight,
                         -np.sqrt(1.0 / interm_features),
                          np.sqrt(1.0 / interm_features))
        self.filter = _make_impsq_filter(filter_type, in_features, interm_features, scale_per_layer, **opts)

    def forward(self, x, z):
        return self.filter(x, self.z_map(z))


class _ImpSqMFNGeneric(nn.Module):
    """Inner cell f(x, z) = init_filter(x) + L_n( ... L_1(z) )."""
    def __init__(self, in_features, interm_features, n_layers,
                 input_scale=256.0, filter_type='fourier', **opts):
        super().__init__()
        layer_scale = float(np.sqrt(n_layers + 1))
        self.init_filter = _make_impsq_filter(filter_type, in_features, interm_features,
                                                input_scale / layer_scale, **opts)
        self.layers = nn.ModuleList([
            _ImpSqMFNLayerGeneric(in_features, interm_features, input_scale / layer_scale,
                                  filter_type=filter_type, **opts)
            for _ in range(n_layers)
        ])
        self.register_buffer('init_filter_in', torch.ones(1, interm_features))
        self.interm_features = interm_features

    def forward(self, x, z):
        gx = self.init_filter(x, self.init_filter_in)
        z = z + gx
        for layer in self.layers:
            z = layer(x, z)
        return z


def _build_impsq(in_channels, hidden_channels, out_channels, n_layers,
                 input_scale, max_iter, tol, output_act, filter_type, **opts):
    """Helper used by ImpSq / iMFN / iGabor / iSIREN registered models."""
    class _ImpSqGeneric(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = _ImpSqMFNGeneric(in_channels, hidden_channels, n_layers,
                                       input_scale=input_scale, filter_type=filter_type, **opts)
            self.output_map = nn.Linear(hidden_channels, out_channels)
            self.max_iter = max_iter
            self.tol = tol
            self.output_act = output_act
            self.interm_channels = hidden_channels

        def forward(self, x):
            z0 = torch.zeros(*x.shape[:-1], self.interm_channels, device=x.device, dtype=x.dtype)
            with torch.no_grad():
                z = z0
                for _ in range(self.max_iter):
                    z_new = self.f(x, z)
                    num = (z_new - z).norm().item()
                    den = z.norm().item() + 1e-9
                    z = z_new
                    if num / den < self.tol:
                        break
            z = z.requires_grad_()
            z_out = self.f(x, z)
            out = self.output_map(z_out)
            if self.output_act:
                out = torch.sin(out)
            return out
    return _ImpSqGeneric()


@register('iMFN')
class iMFN(nn.Module):
    """(Implicit)^2 with Fourier inner filter — equivalent to our `ImpSq`."""
    def __init__(self, in_channels=2, hidden_channels=256, out_channels=3,
                 n_layers=4, input_scale=256.0, max_iter=30, tol=1e-3,
                 output_act=False, **_unused):
        super().__init__()
        self._impl = _build_impsq(in_channels, hidden_channels, out_channels,
                                   n_layers, input_scale, max_iter, tol, output_act,
                                   filter_type='fourier')
        self.f = self._impl.f
        self.output_map = self._impl.output_map
        self.interm_channels = self._impl.interm_channels
        self.max_iter = self._impl.max_iter
        self.tol = self._impl.tol

    def forward(self, x):
        return self._impl(x)


@register('iGabor')
class iGabor(nn.Module):
    """(Implicit)^2 with Gabor inner filter."""
    def __init__(self, in_channels=2, hidden_channels=256, out_channels=3,
                 n_layers=4, input_scale=256.0, max_iter=30, tol=1e-3,
                 output_act=False, alpha=5.0, beta=1.0, **_unused):
        super().__init__()
        self._impl = _build_impsq(in_channels, hidden_channels, out_channels,
                                   n_layers, input_scale, max_iter, tol, output_act,
                                   filter_type='gabor', alpha=alpha, beta=beta)
        self.f = self._impl.f
        self.output_map = self._impl.output_map
        self.interm_channels = self._impl.interm_channels
        self.max_iter = self._impl.max_iter
        self.tol = self._impl.tol

    def forward(self, x):
        return self._impl(x)


@register('iSIREN')
class iSIREN(nn.Module):
    """(Implicit)^2 with SIREN-like inner filter (additive: out = sin(z + W x + b))."""
    def __init__(self, in_channels=2, hidden_channels=256, out_channels=3,
                 n_layers=4, input_scale=256.0, max_iter=30, tol=1e-3,
                 output_act=False, **_unused):
        super().__init__()
        self._impl = _build_impsq(in_channels, hidden_channels, out_channels,
                                   n_layers, input_scale, max_iter, tol, output_act,
                                   filter_type='siren_like')
        self.f = self._impl.f
        self.output_map = self._impl.output_map
        self.interm_channels = self._impl.interm_channels
        self.max_iter = self._impl.max_iter
        self.tol = self._impl.tol

    def forward(self, x):
        return self._impl(x)


@register('ImpSq')
class ImpSq(nn.Module):
    """(Implicit)^2 INR (single-cell DEQ over an MFN-Fourier inner f).

    Forward iteration solver to a tolerance, then a single backprop pass through
    the final residual (the standard "1-step" DEQ backward — easier than the
    full Anderson/Broyden adjoint and empirically close on overfit tasks).
    """
    def __init__(self, in_channels=2, hidden_channels=256, out_channels=3,
                 n_layers=4, input_scale=256.0,
                 max_iter=30, tol=1e-3, output_act=False, **_unused):
        super().__init__()
        self.f = _ImpSqMFN(in_channels, hidden_channels, n_layers, input_scale=input_scale)
        self.output_map = nn.Linear(hidden_channels, out_channels)
        self.max_iter = max_iter
        self.tol = tol
        self.output_act = output_act
        self.interm_channels = hidden_channels

    def _forward_iter(self, x, z0):
        """Run forward iteration z_{k+1} = f(x, z_k) until rel-change < tol or max_iter."""
        z = z0
        for _ in range(self.max_iter):
            z_new = self.f(x, z)
            num = (z_new - z).norm().item()
            den = z.norm().item() + 1e-9
            z = z_new
            if num / den < self.tol:
                break
        return z

    def forward(self, x):
        z0 = torch.zeros(*x.shape[:-1], self.interm_channels, device=x.device, dtype=x.dtype)
        # Solve fixed point under no-grad
        with torch.no_grad():
            z_star = self._forward_iter(x, z0)
        # One final differentiable pass for the implicit-grad approximation (1-step backward).
        z_star = z_star.requires_grad_()
        z_out = self.f(x, z_star)
        out = self.output_map(z_out)
        if self.output_act:
            out = torch.sin(out)
        return out


@register('Gauss_Rec')
class Gauss_Rec(nn.Module):
    """Weight-tied recurrent Gauss MLP."""
    def __init__(self, in_channels=2, hidden_channels=256, out_channels=3,
                 hidden_layers=5, scale=30.0, **_unused):
        super().__init__()
        self.n_recs = hidden_layers
        self.input_layer = _GaussLayer(in_channels, hidden_channels, scale=scale)
        self.rec_layer = _GaussLayer(hidden_channels, hidden_channels, scale=scale)
        self.final_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, coords):
        x = self.input_layer(coords)
        for _ in range(self.n_recs):
            x = self.rec_layer(x)
        return self.final_layer(x)


@register('PEMLP_Rec')
class PEMLP_Rec(nn.Module):
    """Weight-tied recurrent PEMLP: PE -> 1 input Linear+ReLU -> N* (1 weight-tied Linear+ReLU) -> out."""
    def __init__(self, in_channels=2, hidden_channels=256, out_channels=3,
                 num_layers=5, N_freqs=10, **_unused):
        super().__init__()
        self.encoding = _PositionalEncoding(in_channels=in_channels, N_freqs=N_freqs)
        self.input_linear = nn.Linear(self.encoding.out_channels, hidden_channels)
        self.rec_linear = nn.Linear(hidden_channels, hidden_channels)
        self.relu = nn.ReLU(True)
        self.final_linear = nn.Linear(hidden_channels, out_channels)
        self.n_recs = num_layers

    def forward(self, coords):
        x = self.relu(self.input_linear(self.encoding(coords)))
        for _ in range(self.n_recs):
            x = self.relu(self.rec_linear(x))
        return self.final_linear(x)


@register('WIRE_Rec')
class WIRE_Rec(nn.Module):
    """Weight-tied recurrent variant of WIRE: 1 input ComplexGaborLayer + 1
    weight-tied recurrent ComplexGaborLayer applied hidden_layers times +
    1 final complex linear -> .real."""
    def __init__(self, in_channels=2, out_channels=3, hidden_channels=256, hidden_layers=5,
                 omega_0={'first': 20., 'hidden': 20.}, scale=10.0, **_unused):
        super().__init__()
        h = int(hidden_channels / np.sqrt(2))
        self.n_recs = hidden_layers
        self.input_layer = _ComplexGaborLayer(in_channels, h, is_first=True,
                                                omega0=omega_0['first'], sigma0=scale)
        self.rec_layer = _ComplexGaborLayer(h, h, omega0=omega_0['hidden'], sigma0=scale)
        self.final_layer = nn.Linear(h, out_channels, dtype=torch.cfloat)

    def forward(self, coords):
        x = self.input_layer(coords)
        for _ in range(self.n_recs):
            x = self.rec_layer(x)
        return self.final_layer(x).real
