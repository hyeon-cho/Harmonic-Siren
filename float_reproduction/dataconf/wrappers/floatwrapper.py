import numpy as np
import torch
from einops import rearrange
from torch.nn import functional as F

from dataconf import register
from dataconf.wrappers.basewrapper import MINIBase


@register('mini-float')
class MINIFloat(MINIBase):
    """
    Float-image wrapper:
    - keeps RGB values as continuous float channels (no bit-plane decomposition)
    - returns rgb shape: [num_query, 3]
    """

    def get_float_rgb(self, img: torch.Tensor) -> torch.Tensor:
        return rearrange(img, 'c h w -> (h w) c')

    def sampling(self, coord: torch.Tensor, sr_coord: torch.Tensor, rgb_float: torch.Tensor) -> dict:
        sample_lst = None
        if self.sample_q > 0 and not self.valid:
            sample_lst = np.random.choice(len(coord), self.sample_q, replace=False)
            coord = coord[sample_lst]
            rgb_float = rgb_float[sample_lst]
            sr_coord = sr_coord[sample_lst]

        return {
            'coord': coord,
            'sr_coord': sr_coord,
            'rgb_float': rgb_float,
            'sample_lst': sample_lst,
        }

    def __getitem__(self, idx):
        img = (self.dataset[idx])[:3]
        img, h, w = self.data_augmentation(img)

        self.rpp = False
        if self.rpp:
            img = self.make_random_permute(img, permutation='random')

        coordinate = self.generate_coord(img)
        sr_coordinate = rearrange(
            F.interpolate(
                rearrange(coordinate, '(h w) xy -> () xy h w', h=h),
                scale_factor=self.scale_factor,
                mode='bilinear',
            ).squeeze(0),
            'xy h w -> (h w) xy',
        )
        rgb_float = self.get_float_rgb(img)
        sampled_batch = self.sampling(coordinate, sr_coordinate, rgb_float)

        if self.bipolar_out:
            rgb_out = sampled_batch['rgb_float'] * 2.0 - 1.0
        else:
            rgb_out = sampled_batch['rgb_float']

        self.first_logging = False
        return {
            'coord': sampled_batch['coord'],
            'rgb': rgb_out,
            'inp_size': torch.Tensor([h, w]),
            'sr_coord': sampled_batch['sr_coord'],
        }
