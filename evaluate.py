# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 20:46:43 2021

@author: leisir
"""

"""
Video Quality Metrics
Copyright (c) 2014 Alex Izvorski <aizvorski@gmail.com>

"""

import numpy
from scipy.ndimage import gaussian_filter

from numpy.lib.stride_tricks import as_strided as ast

from skimage import io

import numpy as np

"""
Hat tip: http://stackoverflow.com/a/5078155/1828289
"""
def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
    strides = (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)

def normalize(img):
      max_i = img.max()
      min_i = img.min()
      
      norm_img = (img - min_i)/(max_i - min_i)
      
      return norm_img

def ssim(img1, img2, C1=0.01**2, C2=0.03**2):

    #bimg1 = block_view(img1, (4,4))
    #bimg2 = block_view(img2, (4,4))
    bimg1 = img1
    bimg2 = img2
    s1  = numpy.sum(bimg1, (-1, -2))
    s2  = numpy.sum(bimg2, (-1, -2))
    ss  = numpy.sum(bimg1*bimg1, (-1, -2)) + numpy.sum(bimg2*bimg2, (-1, -2))
    s12 = numpy.sum(bimg1*bimg2, (-1, -2))

    vari = ss - s1*s1 - s2*s2
    covar = s12 - s1*s2

    ssim_map =  (2*s1*s2 + C1) * (2*covar + C2) / ((s1*s1 + s2*s2 + C1) * (vari + C2))
    return numpy.mean(ssim_map)

# FIXME there seems to be a problem with this code
def ssim_exact(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):

    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    return numpy.mean(ssim_map)

def generate_mask(inL, maskR):
    '''
    Generates a negative central radial mask to conceal a specified area in the OD image
    :param inL: int, UNETs input image size [px]
    :param maskR: int, the mask radius [px]
    :return: binary np.array, negative central radial mask of inLXinL size and blackened circle of maskR radius
    '''
    
    scale = np.arange(inL)
    mask = np.zeros((inL, inL))
    mask[(scale[np.newaxis, :] - (inL - 1) / 2) ** 2 + (scale[:, np.newaxis] - (inL - 1) / 2) ** 2 > maskR ** 2] = 1

    return mask

X = './result/000001.tif'
predict_X = './result/000001_prediction.tif'
ref = './result/000001_ref.tif'

fig_X = io.imread(X)
fig_predict = io.imread(predict_X)
fig_ref = io.imread(ref)

norm_X = normalize(fig_X)[143:333,143:333]
norm_pre = normalize(fig_predict)
norm_ref = normalize(fig_ref)[143:333,143:333]
mask_X = generate_mask(190, 95)

score = ssim(norm_pre*(1-mask_X), norm_X*(1-mask_X))
print('SSIM: ', score)

