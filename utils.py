from torch import nn as nn
from torch.nn import functional as F
from skimage.io import imread
from skimage import img_as_float
from skimage.transform import resize
from skimage.measure import compare_psnr
from skimage.color import rgb2gray, rgb2ycbcr, ycbcr2rgb
from torch.utils.data import Dataset
import os

def normalize_img(x):
    x /= 255
    return img_as_float(x)

def gaussian_kernel(size=3, sig=1.):

    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel / np.sum(kernel)

def make_kernel(kernels='kernels/1d/'):
    psfs = os.listdir(kernels)
    idx = np.random.randint(len(psfs))
    
    psf = imread(kernels + psfs[idx])
    psf /= psf.sum()
    return psf

def blur(x, psf):
    if x.ndim == 2:
        x = fftconvolve(x, psf, mode='same')
    elif x.ndim == 3:
        x[...,0] = fftconvolve(x[...,0], psf, mode='same')
        x[...,1] = fftconvolve(x[...,1], psf, mode='same')
        x[...,2] = fftconvolve(x[...,2], psf, mode='same')
    return x

def reassemble_rgb(recovered_luminance, ycbcr):
    if not isinstance(recovered_luminance, np.ndarray):
        recovered_luminance = recovered_luminance.clone().numpy()
        
    if not isinstance(ycbcr, np.ndarray):
        ycbcr = ycbcr.clone().numpy()
    
    while recovered_luminance.ndim != 2:
        recovered_luminance = recovered_luminance[0,...]
        
    while ycbcr.ndim != 3:
        ycbcr = ycbcr[0,...]
        
    recovered = np.zeros(ycbcr.shape)
    
    recovered[...,0] = recovered_luminance
    recovered[...,1] = normalize_img(ycbcr[...,1])
    recovered[...,2] = normalize_img(ycbcr[...,2])
    
    recovered_rgb = ycbcr2rgb(recovered*255).clip(0,1)
    return img_as_float(recovered_rgb)

class microcells(Dataset):
    def __init__(self, folder='../dsbowl/train/', crop=True):
        
        self.folder = folder
        self.pathes = os.listdir(self.folder)
        
        self.crop = crop
        
    def get_crop_indexes(self, x):
        i = np.random.randint(x.shape[0]-255)
        j = np.random.randint(x.shape[1]-255)
        return i,j
            
    def __len__(self):
        return len(self.pathes)
    
    def __getitem__(self, index):
        
        image_path = self.folder + self.pathes[index] + '/images/'
        image_id = os.listdir(image_path)[0]
        image_path = image_path + image_id
        
        ground_truth = img_as_float(imread(image_path)[...,:3])
        
        psf = make_kernel()
        blurred = blur(ground_truth.copy(), psf=psf)
        
        psf = torch.FloatTensor(psf)[None,...]
        
        blurred_ycbcr = rgb2ycbcr(blurred)
        blurred_ycbcr[...,0] = normalize_img(blurred_ycbcr[...,0])
        
        gt_ycbcr = rgb2ycbcr(ground_truth)
        gt_ycbcr[...,0] = normalize_img(gt_ycbcr[...,0])
        
        if self.crop:
            i,j = self.get_crop_indexes(blurred_ycbcr)
            blurred_ycbcr = blurred_ycbcr[i:i+256,j:j+256,:]
            gt_ycbcr = gt_ycbcr[i:i+256,j:j+256,:]
        
        blurred_ycbcr = torch.FloatTensor(blurred_ycbcr)[None,...]
        gt_ycbcr = torch.FloatTensor(gt_ycbcr)[None,...]

        return blurred_ycbcr, gt_ycbcr, psf