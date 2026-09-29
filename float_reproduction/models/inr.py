import torch.nn as nn

from models.modules import LinearWithMetrics, SineLayer
from models.models import register
from models.spectral_metrics import SpectralAnalysisMixin


@register('MININR-ol')
class OursINR(SpectralAnalysisMixin, nn.Module):
    def __init__(self,
                 in_channels:         int = 3,
                 hidden_channels:     int = 256,
                 out_channels:        int = 3,
                 bias:                bool = False,

                 recurrent_steps:     int = 1, 
                 omega_0:             dict={'first': 30., 'hidden': 30.}):
        super().__init__()
        self.in_channels     = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels    = out_channels
        self.recurrent_steps = recurrent_steps

        self.in_layer      = SineLayer(in_features=in_channels,
                                       out_features=hidden_channels,
                                       bias=bias,
                                       is_first=True,
                                       layer_type='first',
                                       omega_0=omega_0['first'])
        self.recurrnet_net = SineLayer(in_features=hidden_channels,
                                       out_features=hidden_channels,
                                       bias=bias,
                                       is_first=False,
                                       layer_type='hidden',
                                       omega_0=omega_0['hidden'])
        self.out_layer     = LinearWithMetrics(hidden_channels, out_channels, bias=False, layer_type='last')

    def forward(self, x):
        h = self.get_in_features(x)
        h = self.get_hidden_features(h)
        output = self.get_out_features(h)
        return output

    def get_in_features(self, x):
        in_feat = self.in_layer(x)
        return in_feat

    def get_hidden_features(self, in_feat):
        feat = in_feat
        for i in range(self.recurrent_steps):
            feat = self.recurrnet_net(feat) 
        return feat

    def get_out_features(self, hid_feat):
        output = self.out_layer(hid_feat)
        return output

    def get_layer_infos(self):
        infos = [self.in_layer.get_info()]
        shared_info = self.recurrnet_net.get_info()
        infos.extend(dict(shared_info) for _ in range(self.recurrent_steps))
        infos.append(self.out_layer.get_info())
        return infos

    def get_end_to_end_spectral_bound(self):
        input_info = self.in_layer.get_info()
        shared_info = self.recurrnet_net.get_info()
        output_info = self.out_layer.get_info()
        total = input_info['combined_spectral_norm']
        total *= shared_info['combined_spectral_norm'] ** self.recurrent_steps
        total *= output_info['combined_spectral_norm']
        return total

    def get_detailed_matrix_info(self):
        infos = self.get_layer_infos()
        return {
            'layer_infos': infos,
            'end_to_end_spectral_bound': self.get_end_to_end_spectral_bound(),
            'recurrent_steps': self.recurrent_steps,
            'omega_values': [float(self.in_layer.omega_0), float(self.recurrnet_net.omega_0)],
            'num_layers': len(infos),
        }

    def _get_jacobian_layers(self):
        return [self.in_layer, self.recurrnet_net, self.out_layer]

    def get_recurrent_analysis_schedule(self):
        return ['full' for _ in range(self.recurrent_steps)]


