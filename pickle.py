#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:46:00 2017

@author: no1
"""
import pickle
import matplotlib.pyplot as plt
    
def view_samples(epoch, samples):
    """
    epoch代表第几次迭代的图像
    samples为我们的采样结果
    """
    fig, axes = plt.subplots(figsize=(7,7), nrows=5, ncols=5, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch][1]): # 这里samples[epoch][1]代表生成的图像结果，而[0]代表对应的logits
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
    
    return fig, axes

with open('train_samples.pkl', 'rb') as f:
    samples = pickle.load(f)
    
    
view_samples(400, samples)