# Functionality Tested at 2024-08-31
import torch 
from einops import rearrange

from utils._discretization_helper import gray_to_binary_1003, binary_to_gray_1003

""" Transform the RGB Float-point tensor into the Binary-form tensor 
Input: FloatTensor in shape:  [Batch, RGB, Height, Width] in range (0, 1)
Output: FloatTensor in shape: [Batch, Bit, RGB, Height, Width] in range (0, 1)
"""
def batchwise_bit_decomposition(tensor, bit=8):
    _bit_mul = ((2 ** bit) - 1)
    tensor = tensor.clone().mul_(_bit_mul)
    tensor = tensor.type(torch.int8) 
    tensor = torch.stack([(tensor >> i) & 1 for i in range(bit)], dim=1)
    return tensor.float()


""" Transform the Binary-form tensor into the RGB Float-point tensor 
Input:  FloatTensor in shape:  [Batch, Bit, RGB, Height, Width] in range (0, 1)
Output: FloatTensor in shape:  [Batch, RGB, Height, Width] in range (0, 1)
"""
def batchwise_bit_composition(tensor):
    """ Recompose a batch of images from their bitwise components. """
    r = [tensor[:, i, :, :, :] * (2**i) for i in range(8)] 
    r = sum(r).float()
    r /= 255.
    return r


""" Transform the RGB Float-point tensor into the Binary-form tensor 
Input: FloatTensor in shape:  [Batch, RGB, Height, Width] in range (0, 1)
Output: FloatTensor in shape: [Batch, Bit, RGB, Height, Width] in range (0, 1)

limit = -1 
: Just return the original input tensor 
"""
def partial_bit_decomposition(tensor, bit=8, limit=4, float_value_normalize=False):
    assert bit > limit + 1, 'Use standard bit decomposition for this case'
    bit_tensor = batchwise_bit_decomposition(tensor, bit)

    _temporal = rearrange(bit_tensor, 'batch bit rgb h w -> bit batch rgb h w')
    fmsb = 0. 
    cell = torch . zeros_like(_temporal)[:(limit + 1) + 1]
    for i in range ( _temporal . shape [0] - 1, limit, -1):  # e.g., use 7, 6 and 5 bits
        fmsb = fmsb + _temporal[i]*(2**i)
    cell[-1] = fmsb / ((2**bit)-1. ) if float_value_normalize else fmsb
    cell[0:limit+1] = _temporal[0:limit+1]

    # device same as the input tensor
    cell   = cell.to(tensor.device)
    tensor = None 
    return rearrange(cell, 'bit batch rgb h w -> batch bit rgb h w') 


""" Transform the Binary-form tensor into the RGB Float-point tensor 
Input:  FloatTensor in shape:  [Batch, Bit, RGB, Height, Width] in range (0, 1)
Output: FloatTensor in shape:  [Batch, RGB, Height, Width] in range (0, 1)
"""
def partial_bit_composition(tensor, bit=8, float_value_normalized=False):
    _temporal = rearrange(tensor, 'batch bit rgb h w -> bit batch rgb h w')
    _temporal[-1] = _temporal[-1] * ((2**bit)-1.) if float_value_normalized else _temporal[-1]
    _value    = 0. 
    for i in range (_temporal.shape[0] - 1):
        _value = _value + _temporal[i]*(2**i)
    _value = _value + _temporal[-1]
    _value . to(tensor.device); tensor = None
    return _value / 255.


class BiTransform():
    def __init__(self, bit=8):
        self.bit = bit

    def decomposition(self, tensor):
        return batchwise_bit_decomposition(tensor, bit=8)

    def composition(self, tensor):
        return batchwise_bit_composition(tensor)


class GrayTransform():
    def __init__(self, bit=8):
        self.bit = bit

    def decomposition(self, tensor):
        bin_tensor  = batchwise_bit_decomposition(tensor, self.bit)
        gray_tensor = binary_to_gray_1003(bin_tensor) 
        return gray_tensor   #   [:, 1:, ...]

    def composition(self, tensor): 
        gray_tensor = gray_to_binary_1003(tensor)
        composed    = batchwise_bit_composition(gray_tensor)
        return composed
