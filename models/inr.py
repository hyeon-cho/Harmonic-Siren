import torch.nn as nn

from models.modules import SineLayer
from models.models import register


@register('MININR-ol')
class OursINR(nn.Module): 
    def __init__(self,
                 in_channels:         int = 3,
                 hidden_channels:     int = 256,
                 out_channels:        int = 3,
                 bias:                bool = False,

                 recurrent_steps:     int = 1,
                 omega_0:             dict={'first': 30., 'hidden': 30.},

                 # per-layer bias overrides for the bias ablation; when None they
                 # fall back to the paper default (in/rec follow `bias`, out off).
                 bias_in:             bool = None,
                 bias_rec:            bool = None,
                 bias_out:            bool = None):
        super().__init__()
        self.in_channels     = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels    = out_channels
        self.recurrent_steps = recurrent_steps

        bias_in  = bias  if bias_in  is None else bias_in
        bias_rec = bias  if bias_rec is None else bias_rec
        bias_out = False if bias_out is None else bias_out

        self.in_layer      = SineLayer(in_features=in_channels,
                                       out_features=hidden_channels,
                                       bias=bias_in,
                                       is_first=True,
                                       omega_0=omega_0['first'])
        self.recurrnet_net = SineLayer(in_features=hidden_channels,
                                       out_features=hidden_channels,
                                       bias=bias_rec,
                                       is_first=False,
                                       omega_0=omega_0['hidden'])
        self.out_layer     = nn.Linear(hidden_channels, out_channels, bias=bias_out)

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
        for _ in range(self.recurrent_steps):
            feat = self.recurrnet_net(feat)
        return feat

    def get_out_features(self, hid_feat):
        output = self.out_layer(hid_feat)
        return output  


