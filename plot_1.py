# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 14:19:00 2021

@author: leisir
"""

import numpy as np
import matplotlib.pyplot as plt

import cv2
import imageio
import sys
from tqdm import tqdm
from skimage import io

def plot_single_comparison(X, Y_tag, Y, ref):
    '''
    Plots a single comparison between the prediction and target
    :param X: np.array, the masked input image
    :param Y_tag: np.array, the predicted image
    :param Y: np.array, the target image (or the unmasked input image)
    :param ref: np.array, the reference image
    :return:
    '''
    
    fig = plt.figure(figsize=(15, 15))
    fig.set_facecolor('white')
    plt.subplot(2, 2, 1)
    plt.imshow(X, vmin=.1, vmax=.9)
    plt.title('input')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(Y_tag, vmin=.1, vmax=.9)
    plt.title('prediction')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.imshow(Y, vmin=.1, vmax=.9)
    plt.title('target')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.imshow(Y - ref, vmin=-.1, vmax=.1)
    plt.title('target-prediction')
    plt.colorbar()

    return


def plot_log(train_loss, val_loss, referenceMSE):
    '''
    Plots the log curve of the loss function
    :param train_loss: array or array-like object, training loss
    :param val_loss: array or array-like object, validation loss
    :param referenceMSE: array or array-like object, containing the reference MSE for the experiment
    :return:
    '''
    
    plt.semilogy(train_loss, label='train')
    plt.semilogy(val_loss, label='val')
    if referenceMSE is not None:
        plt.semilogy(val_loss * 0 + referenceMSE, label='ref')
        plt.semilogy(val_loss * 0 + referenceMSE *np.pi /4, label='ref*pi/4')
        plt.semilogy(val_loss * 0 + referenceMSE / 2, label='ref/2')

    return


def plot_log_log(train_loss, val_loss, referenceMSE):
    '''
    Plots the log-log curve of the loss function
    :param train_loss: array or array-like object, training loss
    :param val_loss: array or array-like object, validation loss
    :param referenceMSE: array or array-like object, containing the reference MSE for the experiment
    :return:
    '''
    
    plt.loglog(train_loss, label='train')
    plt.loglog(val_loss, label='val')
    if referenceMSE is not None:
        plt.loglog(val_loss * 0 + referenceMSE, label='ref')
        plt.loglog(val_loss * 0 + referenceMSE *np.pi /4, label='ref*pi/4')
        plt.loglog(val_loss * 0 + referenceMSE / 2, label='ref/2')

    return


def plot_runtime_error(epochNum, train_loss, val_loss, referenceMSE):
    '''
    Plots the continuous loss for the training loop
    :param epochNum: int, the current epoch number
    :param train_loss: array or array-like object, training loss
    :param val_loss: array or array-like object, validation loss
    :param referenceMSE: array or array-like object, containing the reference MSE for the experimen
    :return:
    '''
    
    fig = plt.figure(1)
    plt.clf()
    fig.set_facecolor('white')

    if epochNum < 50:
        plot_log(train_loss, val_loss, referenceMSE)
    else:
        plot_log_log(train_loss, val_loss, referenceMSE)

    plt.legend()
    plt.show(block=False)
    plt.pause(0.001)

    return

X = './result/000001.tif'
predict_X = './result/000001_prediction.tif'
ref = './result/000001_ref.tif'

fig_X = io.imread(X)
fig_predict = io.imread(predict_X)
fig_ref = io.imread(ref)

OD_image = np.array(np.array(io.imread(X)), dtype=np.dtype('float32'))

def normalize(img):
      max_i = img.max()
      min_i = img.min()
      
      norm_img = (img - min_i)/(max_i - min_i)
      
      return norm_img

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

OD_norm = normalize(OD_image)

norm_X = normalize(fig_X)[143:333,143:333]
norm_pre = normalize(fig_predict)
norm_ref = normalize(fig_ref)[143:333,143:333]
mask_X = generate_mask(190, 95)

#plt.figure('F')
#sample = np.ones((512,512,3))
#plt.imshow(OD_norm)
#plt.show()

plot_single_comparison(norm_X*mask_X,norm_pre,norm_X,norm_ref)

