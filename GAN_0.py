#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:55:50 2017

@author: no1
"""

import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/')
img_size = mnist.train.images[0].shape[0]
noise_size = 100
g_units = 128
d_units = 128
alpha = 0.01
learning_rate = 0.001
smooth = 0.1
batch_size = 128
epochs = 500
n_sample = 25

class GAN():

    def __init__(self,img_size,noise_size):
        self.real_img = tf.placeholder(tf.float32, [None, img_size], name='real_img')
        self.noise_img = tf.placeholder(tf.float32, [None, noise_size], name='noise_img')
        
    @staticmethod  
    def get_generator(self,noise_img, n_units, out_dim, reuse=False, alpha=0.01):
        with tf.variable_scope("generator", reuse=reuse):
            # hidden layer
            hidden1 = tf.layers.dense(noise_img, n_units)
            # leaky ReLU
            hidden1 = tf.maximum(alpha * hidden1, hidden1) #Leaky ReLU
            # dropout
            hidden1 = tf.layers.dropout(hidden1, rate=0.2)
    
            # logits & outputs
            logits = tf.layers.dense(hidden1, out_dim)
            outputs = tf.tanh(logits)
            return logits,outputs
    @staticmethod
    def get_discriminator(self,img, n_units, reuse=False, alpha=0.01):
    
        
        with tf.variable_scope("discriminator", reuse=reuse):
            # hidden layer
            hidden1 = tf.layers.dense(img, n_units)
            hidden1 = tf.maximum(alpha * hidden1, hidden1)
            
            # logits & outputs
            logits = tf.layers.dense(hidden1, 1)
            outputs = tf.sigmoid(logits)
            return logits,outputs
    @staticmethod
    def view_samples(self, samples):
        """
        epoch代表第几次迭代的图像
        samples为我们的采样结果
        """
        fig, axes = plt.subplots(figsize=(7,7), nrows=5, ncols=5, sharey=True, sharex=True)
        for ax, img in zip(axes.flatten(), samples): # 这里samples[epoch][1]代表生成的图像结果，而[0]代表对应的logits
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        
        return fig, axes
    def inference(self):

        g_logits, g_outputs = self.get_generator(self,self.noise_img, g_units, img_size)
    
        d_logits_real, d_outputs_real = self.get_discriminator(self,self.real_img,d_units)
        d_logits_fake, d_outputs_fake = self.get_discriminator(self,g_outputs,d_units, reuse=True)
        
    
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, 
                                                                             labels=tf.ones_like(d_logits_real)) * (1 - smooth))
    
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
                                                                             labels=tf.zeros_like(d_logits_fake)))
        # 总体loss
        self.d_loss = tf.add(self.d_loss_real, self.d_loss_fake)
        
        # generator的loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                        labels=tf.ones_like(d_logits_fake)) * (1 - smooth))
    
        train_vars = tf.trainable_variables()
        ## generator中的tensor
        self.g_vars = [var for var in train_vars if var.name.startswith("generator")]
        # discriminator中的tensor
        self.d_vars = [var for var in train_vars if var.name.startswith("discriminator")]
        # optimizer
        d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.d_loss, var_list=self.d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.g_loss, var_list=self.g_vars)
        self.saver = tf.train.Saver(var_list=self.g_vars)
        
        return d_train_opt,g_train_opt
        # 开始训练
    def training(self,d_train_opt,g_train_opt):
        samples = []
        losses = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for e in range(epochs):
                for batch_i in range(mnist.train.num_examples//batch_size):
                    batch = mnist.train.next_batch(batch_size)
                    
                    batch_images = batch[0].reshape((batch_size, 784))
                    # 对图像像素进行scale，这是因为tanh输出的结果介于(-1,1),real和fake图片共享discriminator的参数
                    batch_images = batch_images*2 - 1
                    
                    # generator的输入噪声
                    batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
                    
                    # Run optimizers
                    sess.run(d_train_opt, feed_dict={self.real_img: batch_images, self.noise_img: batch_noise})
                    sess.run(g_train_opt, feed_dict={self.noise_img: batch_noise})
                
                # 每一轮结束计算loss
                train_loss_d = sess.run(self.d_loss, 
                                        feed_dict = {self.real_img: batch_images, 
                                                     self.noise_img: batch_noise})
                # real img loss
                train_loss_d_real = sess.run(self.d_loss_real, 
                                             feed_dict = {self.real_img: batch_images, 
                                                         self.noise_img: batch_noise})
                # fake img loss
                train_loss_d_fake = sess.run(self.d_loss_fake, 
                                            feed_dict = {self.real_img: batch_images, 
                                                         self.noise_img: batch_noise})
                # generator loss
                train_loss_g = sess.run(self.g_loss, 
                                        feed_dict = {self.noise_img: batch_noise})
                
                if e%100==0:
                    
                    print("Epoch {}/{}...".format(e+1, epochs),
                          "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(train_loss_d, train_loss_d_real, train_loss_d_fake),
                          "Generator Loss: {:.4f}".format(train_loss_g))    
                # 记录各类loss值
                losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))
                
                # 抽取样本后期进行观察
                sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))
                gen_samples = sess.run(self.get_generator(self,self.noise_img, g_units, img_size, reuse=True),
                                       feed_dict={self.noise_img: sample_noise})
                samples.append(gen_samples)
                
                self.saver.save(sess, './checkpoints/generator.ckpt')
     
        with open('train_samples.pkl', 'wb') as f:
            pickle.dump(samples, f)
        return losses
    
    def draw_loss(self,losses):

        fig, ax = plt.subplots(figsize=(20,7))
        losses = np.array(losses)
        plt.plot(losses.T[0], label='Discriminator Total Loss')
        plt.plot(losses.T[1], label='Discriminator Real Loss')
        plt.plot(losses.T[2], label='Discriminator Fake Loss')
        plt.plot(losses.T[3], label='Generator')
        plt.title("Training Losses")
        plt.legend()
        
    def draw_samples(self):

        epoch_idx = [0, 5, 10, 20, 40, 60, 80, 100, 150, 250] # 一共300轮，不要越界
        show_imgs = []
        with open('train_samples.pkl', 'rb') as f:
            samples = pickle.load(f)
        for i in epoch_idx:
            show_imgs.append(samples[i][1])
            
        # 指定图片形状
        rows, cols = 10, 25
        fig, axes = plt.subplots(figsize=(30,12), nrows=rows, ncols=cols, sharex=True, sharey=True)
        
        
        for sample, ax_row in zip(show_imgs, axes):
            for img, ax in zip(sample, ax_row):
                ax.imshow(img.reshape((28,28)), cmap='Greys_r')
                ax.axis('off')
            
    def test(self):

        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
            sample_noise = np.random.uniform(-1, 1, size=(25, 100))
            logits,gen_samples = sess.run(self.get_generator(self,self.noise_img, g_units, img_size, reuse=True),
                                   feed_dict={self.noise_img: sample_noise})
            
        self.view_samples(self, gen_samples)
        return gen_samples
global g_samples
gan=GAN(img_size=784,noise_size=100)
d_train_opt,g_train_opt=gan.inference()
#losses=gan.training(d_train_opt,g_train_opt)
#gan.draw_loss(lossess)
gan.draw_samples()
#gen_samples=gan.test()