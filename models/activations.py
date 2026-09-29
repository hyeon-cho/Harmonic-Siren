import torch 
import torch.nn as nn
import torch.nn.functional as F

# Define the activation functions
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class BoundedMish(nn.Module):
    def __init__(self):
        super(BoundedMish, self).__init__()
        self.non_linear = Mish()
    
    def forward(self, x):
        y = torch.where(x > 1, -self.non_linear(-(x - 1)) + 1,
                        torch.where(x < -1, self.non_linear(x + 1) - 1,
                                    x))
        return y

class AperiodicSine(nn.Module):
    def __init__(self):
        super(AperiodicSine, self).__init__()
    
    def forward(self, x):
        return torch.sin(torch.atan(x))
