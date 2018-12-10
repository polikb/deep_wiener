import torch
import numpy as np

from torch import nn as nn
import torch.nn.functional as F

#
def is_complex(x):
    return x.shape[-1] == 2 

def to_complex(x):
    return torch.cat([x[...,None], torch.zeros_like(x[...,None])],-1)

def conjugate(x):
    x[...,1] *= -1
    return x

def real(x):
    return x[...,0]

def imag(x):
    return x[...,1]

def absolute(x):
    Re = torch.pow(real(x).clone(), 2)
    Im = torch.pow(imag(x).clone(), 2)
    return torch.sqrt(Re+Im)
    
def FFT(x, signal_ndim, normalized=False):
    while len(x.shape) <= signal_ndim:
        x = x[None,...]
    return torch.fft(x, signal_ndim, normalized)

def iFFT(x, signal_ndim, normalized=False):
    while len(x.shape) <= signal_ndim:
        x = x[None,...]
    return torch.ifft(x, signal_ndim, normalized)

def irFFT(x, signal_ndim, normalized=False):
    while len(x.shape) <= signal_ndim:
        x = x[None,...]
    return torch.irfft(x, signal_ndim, normalized, onesided=False)

def mat_div(x,y):
    div = x/y
    div[torch.isinf(div)] = 0.
    div[torch.isnan(div)] = 0.
    return div

def complex_div(x,y):
    normalized = absolute(y)
    realx, imagx = real(x).clone(), imag(x).clone()
    realy, imagy = real(y).clone(), imag(y).clone()
    return torch.stack([(realx*realy + imagx*imagy)/normalized, 
                        (imagx*realy - realx*imagy)/normalized], dim = -1)

def complex_multiplication(t1, t2):
    real1, imag1 = real(t1).clone(), imag(t1).clone()
    real2, imag2 = real(t2).clone(), imag(t2).clone()
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)

def get_dim(x):
    assert len(x.shape) in [4,5], "image shoud be Real in format [B, C, W, H] or [B, C, W, H, i]"
    if is_complex(x):
        x = real(x)
    shape = x.shape[-3:]
    if shape[0] in [1,3]:
        return 2
    else:
        print('Uknown dimension')
        return None
    
def get_shape(x, omit_c = True):
    assert len(x.shape) in [4,5], "image shoud be Real in format [B, C, W, H] or [B, C, W, H, i]"
    if is_complex(x):
        x = real(x)
    shape = x.shape[-3:]
    if shape[0] in [1,3]:
        if omit_c:
            return shape[1:]
        else:
            return shape
    else:
        return x.shape[-2:]