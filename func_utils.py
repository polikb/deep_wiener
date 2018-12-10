import torch
from complex_utils import *

def roll(tensor, shift, axis):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)

def get_transfer_func(defining_kernel, shape):
    
    dim = len(defining_kernel.shape)
    defining_kernel_padded = torch.zeros(shape)
    
    defining_kernel_padded[:defining_kernel.shape[0],:defining_kernel.shape[1]] = defining_kernel
    
    #make it periodic
    for axis, axis_size in enumerate(defining_kernel.shape):
        if axis >= len(defining_kernel.shape) - dim:
            
            defining_kernel_padded = roll(defining_kernel_padded,
                               shift=-int(np.floor(axis_size / 2)),
                               axis=axis)
    
    defining_kernel_padded = to_complex(defining_kernel_padded)
    
    return FFT(defining_kernel_padded, len(shape))

def inverse_transfer_func(transfer_func, shape):
    unrolled=real(iFFT(transfer_func, len(transfer_func.shape)))[0]
    
    for axis, axis_size in enumerate(shape):
        unrolled = roll(unrolled, shift=int(np.floor(axis_size / 2)), axis=axis)
        
    unrolled_unpadded = unrolled[:shape[0], :shape[1]]
    return unrolled_unpadded

def roll_batch(tensor, shift, axis):
    
    if shift == 0:
        return tensor
    
    tensor_dim = get_dim(tensor)
    if axis < 0:
        axis += tensor_dim

    dim_size = get_shape(tensor)[axis]
    after_start = dim_size - shift
    
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)
    
    if tensor_dim == 2:
        if axis==0:
            before = tensor[:,:,:dim_size - shift,:]
            after = tensor[:,:,after_start:after_start+shift,:]
        elif axis==1:
            before = tensor[:,:,:,:dim_size - shift]
            after = tensor[:,:,:,after_start:after_start+shift]
        return torch.cat([after, before], axis+2)
    
    elif tensor_dim == 3:
        print("3D not implemented yet")
        return None

def get_transfer_func_batch(defining_kernel, shape):
    
    dim = get_dim(defining_kernel)
    
    defining_kernel_padded = torch.zeros(shape)
    defining_kernel_padded[:,:,:defining_kernel.shape[-2],:defining_kernel.shape[-1]] = defining_kernel
    
    #make it periodic
    for axis, axis_size in enumerate(get_shape(defining_kernel)):
        if axis >= len(get_shape(defining_kernel)) - dim:
            
            defining_kernel_padded = roll_batch(defining_kernel_padded,
                                                shift=-int(np.floor(axis_size / 2)),
                                                axis=axis)
            
    defining_kernel_padded = to_complex(defining_kernel_padded)
    return FFT(defining_kernel_padded, dim)

def inverse_transfer_func_batch(transfer_func, shape):
    kernel_dim = get_dim(transfer_func)
    
    unrolled=real(iFFT(transfer_func, kernel_dim))
    for axis, axis_size in enumerate(shape):
        unrolled = roll_batch(unrolled, shift=int(np.floor(axis_size / 2)), axis=axis)
    
    unrolled_unpadded = unrolled[:,:,:shape[0], :shape[1]]
    return unrolled_unpadded

def manual_norm(x):
    means = x.mean(1).mean(1)
    xc = (x.view(x.shape[0], -1).transpose(0,1) - means)
    xcs = xc/xc.norm(dim=0)
    return xcs.transpose(0,1).view(*x.shape)