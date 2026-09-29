import torch
import random
from torch.utils.data import Dataset

from utils.geometry import make_coord
from utils.discretization import BiTransform, GrayTransform

from einops import rearrange
import numpy as np 

from torchvision.transforms import CenterCrop, Resize

from dataconf import register, make
from torch.nn import functional as F

@register('mini-base')
class MINIBase(Dataset):
    def __init__(self, dataset, 
                 init_down:     int=1,      # Initial Downsample for memory
                 center_crop:   bool=False,
                 resize:        int=256,    # Resize for the image
                 rgb_range:     int=8,
                 coord_config:  dict={
                     'range': [[0, 256], [0, 256]], 
                     'range_division': [1, 1]  #[9, 9] old convention
                 }, sample_q:   int=0, 
                 bipolar_out:   bool=True,
                 transformer:   str='binary',  # or gray
                #  scale_factor:  int=2,
                 sr_config: dict = {
                     'scale_factor': 2, 
                     'patch_size': [256,256]
                 },
                 valid:         bool=False,
                 random_permute:bool=False,
                 int_out:       bool=True,
                 bit_decompose: bool=True):
        self.dataset       = make(dataset)
        self.rgb_range     = rgb_range
        self.coord_config  = coord_config
        self.center_crop   = center_crop
        self.init_down     = init_down
        self.resize        = resize
        self.sample_q      = sample_q
        self.valid         = valid
        self.bipolar_out   = bipolar_out
        self.scale_factor  = sr_config['scale_factor']
        self.bit_decompose = bit_decompose
        self.transformer   = transformer
        self.decomposer    = self._load_decomposer(transformer=transformer) if bit_decompose else None
        self.rpp           = random_permute
        self.int_out       = int_out

        self.first_logging= True

    def __len__(self):
        return len(self.dataset)

    def _load_decomposer(self, transformer):
        if transformer == 'binary':
            return BiTransform(bit=self.rgb_range)
        elif transformer == 'gray':
            return GrayTransform(bit=self.rgb_range)
        else:
            raise ValueError(f"Invalid Transformer: {self.transformer}")

    def data_augmentation(self, img: torch.Tensor) -> torch.Tensor: 
        def _downsample(self, img: torch.Tensor) -> torch.Tensor:
            if self.init_down is not None:
                return Resize([int(img.shape[-2]/self.init_down), 
                            int(img.shape[-1]/self.init_down)])(img)
            return img
        
        def _centre_crop(self, img: torch.Tensor) -> torch.Tensor:
            if self.center_crop: 
                _m  = min(img.shape[-2], img.shape[-1])
                return CenterCrop([_m, _m])(img)
            return img

        def _resize_img(self, img: torch.Tensor) -> torch.Tensor:
            if self.resize is not None:
                return Resize([self.resize, self.resize])(img)
            return img

        img  = _downsample(self, img)
        img  = _centre_crop(self, img)
        img  = _resize_img(self, img)
        h, w = img.shape[-2:]
        if self.first_logging:
            print(f'[Image Augnemtation]\n- Initial downsample for GPU memory: {self.init_down}\n- Center Crop: {self.center_crop}\n- Resize: {self.resize}')
        return img, h, w

    def generate_coord(self, img: torch.Tensor) -> torch.Tensor:
        coord = make_coord(img.shape[-2:], 
                          [(self.coord_config['range'])[0], 
                           (self.coord_config['range'])[1]]
                          )
        coord[:, 0] = coord[:, 0] / self.coord_config['range_division'][0]
        coord[:, 1] = coord[:, 1] / self.coord_config['range_division'][1]

        if self.first_logging:
            print(f'[Coordinate Range]\nX: {coord[:, 0].min():.3f} ~ {coord[:, 0].max():.3f}\nY: {coord[:, 1].min():.3f} ~ {coord[:, 1].max():.3f}')
        return coord

    def get_binarized_rgb(self, img: torch.Tensor) -> torch.Tensor:
        rgb_expand = rearrange(
            self.decomposer.decomposition(rearrange(img, 'c h w -> () c h w')),
            'batch bit rgb h w -> batch (h w) (rgb bit)').squeeze(0)
        return rgb_expand

    def get_binarized_rgb_msbtest(self, img: torch.Tensor) -> torch.Tensor:
        decomposer = self._load_decomposer(transformer=self.transformer)
        rgb_expand = decomposer.decomposition(rearrange(img, 'c h w -> () c h w')).squeeze(0)
        return rgb_expand

    def make_random_permute(self, img: torch.Tensor, 
                            permutation: str='random',
                            seed: int=42) -> torch.Tensor:
        random.seed(seed)
        np.random.seed(seed)

        h = img.shape[1]
        img = rearrange(img, 'c h w -> (h w) c')

        if permutation == 'random':
            encodebook=np.random.permutation(len(img))
            decodebook=np.zeros_like(encodebook)
            for i,val in enumerate(encodebook):
                decodebook[val]=i
            np.save('./decode_book.npy',decodebook)
            enc_img=img[encodebook]

            return rearrange(enc_img, '(h w) c -> c h w', h=h)

        else: 
            raise ValueError(f"Invalid Permutation: {permutation}")

    def sampling(self, coord: torch.Tensor, sr_coord: torch.Tensor, rgb_bin_expand: torch.Tensor) -> dict:
        sample_lst = None 
        if self.sample_q > 0 and not self.valid:
            sample_lst = np.random.choice(len(coord), self.sample_q, replace=False)

            coord      = coord[sample_lst]
            rgb_bin_expand = rgb_bin_expand[sample_lst]
            sr_coord = sr_coord[sample_lst]

        return {
            'coord'          : coord, 
            'sr_coord'       : sr_coord,
            'rgb_bin_expand' : rgb_bin_expand, 
            'sample_lst'     : sample_lst
        }

    def __getitem__(self, idx):
        img            = (self.dataset[idx])[:3]            # img.shape = [C, H, W]
        img, h, w      = self.data_augmentation(img)

        self.rpp = False
        if self.rpp:
            img = self.make_random_permute(img, permutation='random')

        coordinate     = self.generate_coord(img)
        sr_coordinate = rearrange(F.interpolate(rearrange(coordinate, '(h w) xy -> () xy h w', h=h),
                                                scale_factor=self.scale_factor, mode='bilinear').squeeze(0),
                                  'xy h w -> (h w) xy')
        if self.bit_decompose:
            rgb_bin_expand = self.get_binarized_rgb(img)
        else:
            rgb_bin_expand = rearrange(img, 'c h w -> (h w) c')
        sampled_batch  = self.sampling(coordinate, sr_coordinate, rgb_bin_expand)

        if self.bipolar_out:
            rgb_out = sampled_batch['rgb_bin_expand'] * 2 - 1
        else:
            rgb_out = sampled_batch['rgb_bin_expand']

        self.first_logging = False
        return {
            'coord': sampled_batch['coord'],
            'rgb'  : rgb_out,
            'inp_size': torch.Tensor([h, w]), 
            'sr_coord': sampled_batch['sr_coord'],
        }
