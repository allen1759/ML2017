#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 1 10:04:45 2017

@author: Allen
"""

# %%

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


image_dir = 'faceExpressionDatabase'

def PCA(data):
    aver = np.average(data, axis=0)
    arr = data - aver
    U, s, V = np.linalg.svd(arr.T, full_matrices=False)
    S = np.diag(s)
    
    return U, S, V

def CalcCoeff(imgs, aver):
    recons_coeff = []
    for ele in imgs:
        coeff = []
        for i in range(imgs.shape[0]):
            coeff.append(np.inner(ele-aver, U.T[i]))
        coeff = np.array(coeff)
        recons_coeff.append(coeff)
        
    return np.matrix(recons_coeff)

def CalcReconstruct(recons_coeff, aver, k, eigenvec):
    coeff = recons_coeff[:, :k]
    reconstruct = []
    for ele in coeff:
        reconstruct.append(np.array(ele.dot(eigenvec[:k])).flatten() + aver)
    
    return np.array(reconstruct)

def MRSE(origin, reconstruct):
#    rmax = np.max(reconstruct)
#    rmin = np.min(reconstruct)
#    recons = (reconstruct - rmin) * 255.0 / (rmax-rmin)
    
    return np.average((origin - reconstruct)**2)**0.5 / 255.0

# %%

if __name__ == "__main__":

    imgs = []
    for w in range(ord('A'), ord('A') + 10):
        for i in range(1, 11):
            img = Image.open(os.path.join(image_dir, chr(w) + str(i).zfill(2) +'.bmp'))
            arr = np.array(img)
            imgs.append(arr.flatten())
    imgs = np.array(imgs, dtype=np.float32)
    
    U, S, V = PCA(imgs)
    SV = S.dot(V)
    
# %% problem 1.1 left

    aver = np.average(imgs, axis=0)
    img = Image.fromarray(np.uint8(aver.reshape(64,64)))
    img.save('average.png')
    
# %% problem 1.1 right

    fig = plt.figure(figsize=(6, 6), dpi=80)
    
    for i in range(9):
        sub = fig.add_subplot(3, 3, i+1)
        sub.imshow(U.T[i].reshape((64, 64)), cmap='gray')
        sub.axes.get_xaxis().set_ticks([])
        sub.axes.get_yaxis().set_ticks([])
        sub.set_xlabel('#' + str(i) + ' eigenface')
    
    fig.savefig('eigenfaces.png')

# %% problem 1.2

    fig = plt.figure(figsize=(10, 10), dpi=80)    
    for i in range(imgs.shape[0]):
        sub = fig.add_subplot(10, 10, i+1)
        sub.imshow(imgs[i].reshape((64, 64)), cmap='gray')
        sub.axes.get_xaxis().set_ticks([])
        sub.axes.get_yaxis().set_ticks([])
        
    fig.savefig('origin.png')


    recons_coeff = CalcCoeff(imgs, aver)

    k = 1
    reconstruct = CalcReconstruct(recons_coeff, aver, k, U.T)
    
    fig = plt.figure(figsize=(10, 10), dpi=80)
    for i in range(len(reconstruct)):
        sub = fig.add_subplot(10, 10, i+1)
        sub.imshow(reconstruct[i].reshape((64, 64)), cmap='gray')
        sub.axes.get_xaxis().set_ticks([])
        sub.axes.get_yaxis().set_ticks([])
        
    fig.savefig('reconstructk=5.png')
    
# %% problem 1.3

    for k in range(5, 101):
        reconstruct = CalcReconstruct(recons_coeff, aver, k, U.T)
        err = MRSE(imgs, reconstruct)
        print('k={}, MRSE={}'.format(k, err))
        if err < 0.01:
            break;
    
    fig = plt.figure(figsize=(10, 10), dpi=80)
    for i in range(len(reconstruct)):
        sub = fig.add_subplot(10, 10, i+1)
        sub.imshow(reconstruct[i].reshape((64, 64)), cmap='gray')
        sub.axes.get_xaxis().set_ticks([])
        sub.axes.get_yaxis().set_ticks([])
        
    fig.savefig('reconstructk={}.png'.format(k))
