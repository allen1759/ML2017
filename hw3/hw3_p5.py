#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 21:04:45 2017

@author: Allen
"""

# %%

import os
import numpy as np
import keras
from keras import backend as K
import matplotlib.pyplot as plt

# %%
num_classes = 7
img_rows = 48
img_cols = 48

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func):
    """
    Implement this function!
    """
    # run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iter_func([input_image_data, 0])
        input_image_data += grads_value * num_step
    
    return [input_image_data.reshape(img_rows, img_cols), loss_value]
    #return filter_images
    
# %%

if __name__ == "__main__":
     
    filter_dir = 'filter'
    nb_filter = 64
    num_step = 0.8
    
    emotion_classifier = keras.models.load_model('model8.h5')
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
    
    input_img = emotion_classifier.input

    name_ls = ['conv2d_42', 'conv2d_43']
    collect_layers = [layer_dict[name].output for name in name_ls ]

    for cnt, c in enumerate(collect_layers):
        filter_imgs = []
        # filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
        for filter_idx in range(nb_filter):
            # we start from a gray image with some noise
            input_img_data = np.random.random((1, 48, 48, 1)) # random noise
            # compute the gradient of the input picture wrt this loss
            target = K.mean(c[:, :, :, filter_idx])
            # normalization trick: we normalize the gradient
            grads = normalize(K.gradients(target, input_img)[0])
            # this function returns the loss and grads given the input picture
            iterate = K.function([input_img, K.learning_phase()], [target, grads])
            # "You need to implement it."
            filter_imgs.append(grad_ascent(num_step, input_img_data, iterate))
            
        #for it in range(NUM_STEPS//RECORD_FREQ):
        fig = plt.figure(figsize=(14, 8))
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16, 16, i+1)
            ax.imshow(filter_imgs[i][0], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.xlabel('{:.3f}'.format(filter_imgs[i][1]))
            plt.tight_layout()
        fig.suptitle('Filters of layer {})'.format(name_ls[cnt]))
        if not os.path.isdir(filter_dir):
            os.mkdir(filter_dir)
        fig.savefig(os.path.join(filter_dir,'{}'.format(name_ls[cnt])))
        
        
        
        

