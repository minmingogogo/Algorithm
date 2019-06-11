# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:01:28 2019

@author: Scarlett
https://blog.csdn.net/index20001/article/details/74322198
"""
import tensorflow as tf
import numpy as np
import os


#np.newaxis的作用就是在这一位置增加一个一维，这一位置指的是np.newaxis所在的位置，
x_data = np.linspace(-1,1,300)[:,np.newaxis]

noise = np.random.normal(0,0.05,x_data.shape)
#   正太分布 numpy.random.normal(loc=0.0, scale=1.0, size=None)
    #loc : 均值 ；sacle : 标准差 ; size : 输出的shape

y_data = np.square(x_data) - 0.5 + noise

#   输入层
x_ph = tf.placeholder(tf.float32,[None,1])
y_ph = tf.placeholder(tf.float32,[None,1])

#   隐层
w1 = tf.Variable(tf.random_normal([1,10]))
b1 = tf.Variable(tf.zeros([1,10])+0.1)
wx_plus_b1 = tf.matmul(x_ph,w1) + b1
hidden = tf.nn.relu(wx_plus_b1)

#   输出
w2 = tf.Variable(tf.random_normal([10,1]))
b2 = tf.Variable(tf.zeros([1,1]) + 0.1)
wx_plus_b2 = tf.matmul(hidden,w2) + b2
y = wx_plus_b2

#   损失
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ph-y),reduction_indices=[1]))
#   设置学习率
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


#   保存模型对象saver
saver = tf.train.Saver()

#   判断模型保存路径是否存在，不存在则创建
if not os.path.exists('tmp/'):
    os.mkdir('tmp/')
    
#   初始化
    #   gpu 动态申请显存
#    http://www.pianshen.com/article/3365129563/
gpu_options = tf.GPUOptions(allow_growth = True)
config = tf.ConfigProto(gpu_options = gpu_options)
        
with tf.Session(config=config) as sess:
    if os.path.exists('tmp/checkpoint'):        #判断模型是否存在
        saver.restore(sess,'tmp/model.ckpt')    # 从模型中恢复变量
    else:
        init = tf.global_variables_initializer()    #   不存在则初始化变量
        sess.run(init)
    for i in range(1000):
        _,loss_value = sess.run([train_op,loss],feed_dict = {x_ph : x_data, y_ph : y_data})
        if i%50==0:
            save_path = saver.save(sess,'tmp/model.ckpt')
            print("迭代次数 ： %d , 训练损失 ： %s "%(i,loss_value))
        
        























