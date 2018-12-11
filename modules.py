import torch

from torch import nn as nn
from torch.nn.parameter import Parameter

from complex_utils import *
from func_utils import *
from functions import *

class pure_wiener(nn.Module):
    def __init__(self, filter_dim=(5,5), num_filters=24, init_alpha=-2):
        super(pure_wiener, self).__init__()
        
        weight = manual_norm(torch.randn((num_filters, filter_dim[0], filter_dim[1])))
        weight = nn.Parameter(weight, requires_grad=True)
        self.register_parameter('weight', weight)
        
        alpha = torch.FloatTensor([init_alpha])
        alpha = nn.Parameter(alpha, requires_grad=True)
        self.register_parameter('alpha', alpha)
                
    def forward(self, input, psf):
        batch_size = input.shape[0]
        return Wiener2d_func.apply(input, psf, self.weight.expand((batch_size,*self.weight.shape)), self.alpha)
    
class Wiener_block(nn.Module):
    def __init__(self, psf, filter_dim=(5,5), num_filters=24, init_alpha=-2, normalize_filters=True, 
                 clamp=True):
        super(Wiener_block, self).__init__()
        
        self.psf = psf
        self.normalize_filters = normalize_filters
        self.clamp = clamp
        self.num_filters = num_filters
        self.filter_dim = filter_dim
        
        if self.normalize_filters:
            self.wiener = nn.utils.weight_norm(pure_wiener(filter_dim, num_filters, init_alpha), dim=0)
        else:
            self.wiener = pure_wiener(filter_dim, num_filters, init_alpha)
        
    def forward(self, input):
        if self.clamp:
            return self.wiener(input, self.psf).clamp(-1,1)
        
        return self.wiener(input, self.psf)
    
    
class Wiener_dec(nn.Module):
    def __init__(self, filter_dim=(5,5), num_filters=24, init_alpha=-2, normalize_filters=True, 
                 clamp=True):
        super(Wiener_dec, self).__init__()
        
        self.normalize_filters = normalize_filters
        self.clamp = clamp
        self.num_filters = num_filters
        self.filter_dim = filter_dim
        
        if self.normalize_filters:
            self.wiener = nn.utils.weight_norm(pure_wiener(filter_dim, num_filters, init_alpha), dim=0)
        else:
            self.wiener = pure_wiener(filter_dim, num_filters, init_alpha)
        
    def forward(self, input, psf):
        if self.clamp:
            return self.wiener(input, psf).clamp(-1,1)
        
        return self.wiener(input, psf)