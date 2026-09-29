import torch 

# def polarize(tensor, threshold=0, high=1, low=-1): 
#     return torch.where(tensor > threshold, high, low)
class Polarize():
    def __init__(self, threshold=0, high=1, low=-1):
        self.threshold = threshold
        self.high = high
        self.low = low

    def thresholding(self, tensor):
        return torch.where(tensor > self.threshold, self.high, self.low)

    def bypass(self, tensor):
        return tensor

    def clamp(self, tensor):
        return torch.clamp(tensor, self.low, self.high)


class Ternarize():
    def __init__(self):
        pass

    def thresholding(self, tensor):
        tensor = torch.round(tensor)
        return tensor 


class Binarize():
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def thresholding(self, tensor):
        return torch.where(tensor < self.threshold, 0, 1)


class Identity():
    def __init__(self):
        pass

    def thresholding(self, tensor):
        return tensor


class PartialPolarize():
    def __init__(self, threshold=0, high=1, low=-1, bypass_idx=[-1, 5,11,17]):
        self.threshold = threshold
        self.high = high
        self.low = low
        self.bypass_idx = bypass_idx

    def thresholding(self, tensor):
        if self.bypass_idx[0] != -1:
            self.bypass_idx = [-1] + self.bypass_idx

        for idx in range(len(self.bypass_idx)):
            tensor[..., self.bypass_idx[idx-1]+1:self.bypass_idx[idx]] \
                = torch.where(tensor[..., self.bypass_idx[idx-1]+1:self.bypass_idx[idx]] > self.threshold, self.high, self.low)
        return tensor


class Polarize0910():
    def __init__(self, threshold=0, high=1, low=-1):
        self.threshold = threshold
        self.high = high
        self.low = low

    def thresholding(self, tensor):
        tensor[..., :3] = torch.where(tensor[..., :3] > self.threshold, self.high, self.low)
        return tensor
