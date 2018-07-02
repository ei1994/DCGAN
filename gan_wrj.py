# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 21:01:58 2017

@author: DELL
"""

import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/')

#img = mnist.train.images[0]
#plt.imshow(img.reshape((28, 28)), cmap='Greys_r')

# 定义参数
# 真实图像的size
img_size = mnist.train.images[0].shape[0]
# 传入给generator的噪声size
noise_size = 100
# 生成器隐层参数
g_units = 128
# 判别器隐层参数
d_units = 128
# leaky ReLU的参数
alpha = 0.01
# learning_rate
learning_rate = 0.001
# label smoothing
smooth = 0.1
# batch_size
batch_size = 64
# 训练迭代轮数
epochs = 50
# 抽取样本数
n_sample = 25


class GAN():
    def __init__(self, img_size, noise_size):
        self.real_img = tf.placeholder(tf.float32, [None, img_size], name='real_img')
        self.noise_img = tf.placeholder(tf.float32, [None, noise_size], name='noise_img')
        
    @staticmethod    
    def get_generator( noise_img, n_units, out_dim, reuse=False, alpha=0.01):
        """
    生成器
    noise_img: 生成器的输入
    n_units: 隐层单元个数
    out_dim: 生成器输出tensor的size，这里应该为32*32=784
    alpha: leaky ReLU系数
        """
        with tf.variable_scope("generator", reuse=reuse):
            # hidden layer
            hidden1 = tf.layers.dense(noise_img, n_units)
            # leaky ReLU
            hidden1 = tf.maximum(alpha * hidden1, hidden1)
            # dropout
            hidden1 = tf.layers.dropout(hidden1, rate=0.2)
    
            # logits & outputs
            logits = tf.layers.dense(hidden1, out_dim)  # shape=(?, 784)
            outputs = tf.tanh(logits)   #shape=(?, 784)
            
            return logits, outputs
    @staticmethod 
    def get_discriminator( img, n_units, reuse=False, alpha=0.01):
        """
    判别器
    n_units: 隐层结点数量
    alpha: Leaky ReLU系数
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            # hidden layer  全连接层
            hidden1 = tf.layers.dense(img, n_units)
            hidden1 = tf.maximum(alpha * hidden1, hidden1)  # leaky relu 
            
            # logits & outputs
            logits = tf.layers.dense(hidden1, 1)  # 输出判别器的判断，真或假
            outputs = tf.sigmoid(logits)
            
            return logits, outputs
    

    def draw_loss(self,losses):
        
        fig, ax = plt.subplots(figsize=(20,7))
        losses = np.array(losses)  
        plt.plot(losses.T[0], label='Discriminator Total Loss')  # 取loss的每一列画曲线
        plt.plot(losses.T[1], label='Discriminator Real Loss')
        plt.plot(losses.T[2], label='Discriminator Fake Loss')
        plt.plot(losses.T[3], label='Generator')
        plt.title("Training Losses")
        plt.legend()
    
    def inference(self):
        # generator  shape=(?, 784)
        g_logits, g_outputs = GAN.get_generator(self.noise_img, g_units, img_size)
        
        # discriminator  输出判断的结果，真或假
        d_logits_real, d_outputs_real = GAN.get_discriminator(self.real_img, d_units)
        d_logits_fake, d_outputs_fake = GAN.get_discriminator(g_outputs, d_units, reuse=True)
        # 识别真实图片, 让判断1的项更soft, 防止过拟合的方式
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, 
                                                                             labels=tf.ones_like(d_logits_real)) * (1 - smooth))
        # 识别生成的图片
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
                                                                             labels=tf.zeros_like(d_logits_fake)))
        # 总体loss
        self.d_loss = tf.add(self.d_loss_real, self.d_loss_fake)
        
        # generator的loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                        labels=tf.ones_like(d_logits_fake)) * (1 - smooth))
        train_vars = tf.trainable_variables()  # 返回的是需要训练的变量列表

        # generator中的tensor  ，遍历列表，找到以指定单词开头的变量
        self.g_vars = [var for var in train_vars if var.name.startswith("generator")]
        # discriminator中的tensor
        self.d_vars = [var for var in train_vars if var.name.startswith("discriminator")]
        
        # optimizer,,只优化指定的变量，其他保持不变
        d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.d_loss, var_list=self.d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.g_loss, var_list=self.g_vars)
        self.saver = tf.train.Saver(var_list=self.g_vars)
        return d_train_opt,g_train_opt
#    @classmethod    
    def train(self, d_train_opt,g_train_opt):
        
#        cls().inference()
        samples = []
        # 存储loss
        losses = []
        # 开始训练
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
                    _ = sess.run(d_train_opt, feed_dict={self.real_img: batch_images, self.noise_img: batch_noise})
                    _ = sess.run(g_train_opt, feed_dict={self.noise_img: batch_noise})
                
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
                
                    
                print("Epoch {}/{}...".format(e+1, epochs),
                      "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(train_loss_d, train_loss_d_real, train_loss_d_fake),
                      "Generator Loss: {:.4f}".format(train_loss_g))    
                # 记录各类loss值
                losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))
                
                # 抽取样本后期进行观察
                sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))
                gen_samples = sess.run(GAN.get_generator(self.noise_img, g_units, img_size, reuse=True),
                                       feed_dict={self.noise_img: sample_noise})
                samples.append(gen_samples)  # 每轮训练后都保存一下使用数据生成的图片
                
                # 存储checkpoints
                self.saver.save(sess, './checkpoints/generator_1.ckpt')

        with open('train_samples_1.pkl', 'wb') as f:
             pickle.dump(samples, f)
        return losses

    @staticmethod 
    def view_samples( samples):

            fig, axes = plt.subplots(figsize=(7,7), nrows=5, ncols=5, sharey=True, sharex=True)
            for ax, img in zip(axes.flatten(), samples): # 这里samples[epoch][1]代表生成的图像结果，而[0]代表对应的logits
#                ax.xaxis.set_visible(False)
#                ax.yaxis.set_visible(False)
                ax.axis('off')
                ax.imshow(img.reshape((28,28)), cmap='Greys_r')
            
            return fig, axes
        
    def draw_samples(self):

            epoch_idx = [0, 5, 10, 20, 40] # 一共300轮，不要越界
            show_imgs = []
            with open('train_samples_1.pkl', 'rb') as f:
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
            logits,gen_samples = sess.run(GAN.get_generator(self.noise_img, g_units, img_size, reuse=True),
                                   feed_dict={self.noise_img: sample_noise})
            
        GAN.view_samples( gen_samples)
        return gen_samples

gan = GAN(img_size, noise_size)
d_train_opt,g_train_opt=gan.inference()
#losses=gan.train(d_train_opt,g_train_opt)
#gan.draw_loss(losses)
#plt.savefig("losses_2.jpg") 
#gan.draw_samples()
gen_samples=gan.test()


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

