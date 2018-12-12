from torch import nn as nn
from torch.autograd import Variable, Function
import torch.nn.functional as F

from deep_wiener.complex_utils import *
from deep_wiener.func_utils import *

class Wiener2d_func(Function):
    
    @staticmethod
    def forward(ctx, input, psf, filters, alpha):
        #transform input to complex if it's not
        if not is_complex(input):
            input = to_complex(input)

        #setup vars for backward
        ctx.save_for_backward(input, psf, filters, alpha)

        '''do wiener filter with learned parameters'''
        if not is_complex(input):
            input = to_complex(input)

        image_shape = get_shape(input)
        image_dim = get_dim(input)
        full_shape = input.shape[:-1] #real

        D = get_transfer_func_batch(psf, full_shape)
        D_Fy = complex_multiplication(conjugate(D), FFT(input, image_dim))

        #BEGIN: squared sum of regularizers spectrum
        reg_squared_sum_spect = torch.zeros((*filters.shape[:2],*full_shape[-2:]))
        for i in range(filters.shape[1]):
            Dg = absolute(get_transfer_func_batch(filters[:,i,...][:,None,...], full_shape))
            reg_squared_sum_spect[:,i,...] = Dg[:,0,...]**2

        sum_reg = reg_squared_sum_spect.sum(1)[:,None,...]
        #END: squared sum of regularizers spectrum

        L = torch.pow(absolute(D),2) + torch.exp(alpha)*sum_reg
        L = 1/L
        output = irFFT(L[...,None]*D_Fy, image_dim)
        ctx.auxiliary = D, D_Fy, L, sum_reg, image_shape, image_dim, full_shape
                
        return output
    
    @staticmethod
    def backward(ctx, grad_output):

        grad_input, grad_psf, grad_filters, grad_alpha = None, None, None, None
        D, D_Fy, L, sum_reg, image_shape, image_dim, full_shape  = ctx.auxiliary
        input, psf, filters, alpha = ctx.saved_tensors

        '''grad wrt weights'''
        #BEFIN: calculate Z
        L2 = torch.pow(L, 2)
        P = conjugate(FFT(to_complex(grad_output), image_dim))

        D_Fy_Z = complex_multiplication(D_Fy, P)
        Z = L2[...,None]*D_Fy_Z
        #END: calculate Z

        grad_filters = torch.zeros_like(filters)
        one_filter_shape = get_shape(filters)
        for i in range(filters.shape[1]):

            Gj = get_transfer_func_batch(filters[:,i,...][:,None,...], full_shape)
            #BEGIN: calculate each filter's grad
            GjReZ = Gj * real(Z)[...,None]
            grad_brakets = inverse_transfer_func_batch(GjReZ, one_filter_shape)
            grad_filters[:,i,...] = -2*torch.exp(alpha)*grad_brakets[:,0,...]
            #END: calculate each filter's grad

        '''grad wrt aplpha''' #CHECKED::OK
        grad_alpha = (L2*sum_reg*torch.exp(alpha))[...,None]*D_Fy
        grad_alpha = -real(iFFT(grad_alpha, image_dim))*grad_output
        grad_alpha = grad_alpha.sum()[None,...]  #per batch

        '''grad wrt input'''
        grad_input = conjugate(P)
        grad_input = L[...,None]*complex_multiplication(D, P)
        grad_input = irFFT(grad_input, image_dim)

        return grad_input, None, grad_filters, grad_alpha