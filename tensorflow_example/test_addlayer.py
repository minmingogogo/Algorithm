# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:41:54 2019

@author: Scarlett


"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


def add_layer(inputs,in_size,out_size,activation_func = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])) + 0.1
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_func is None:
        outputs  = Wx_plus_b
    else:
        outputs = activation_func(Wx_plus_b)
    return outputs


    
    
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])


l1 = add_layer(xs,1,10,activation_func=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_func=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


prediction_adam = add_layer(l1,10,1,activation_func=None)
loss_adam = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction_adam),reduction_indices=[1]))
train_step_adam = tf.train.AdamOptimizer(0.01).minimize(loss_adam)

#   momentum 参数不会设
#prediction_moment = add_layer(l1,10,1,activation_func=None)
#loss_moment = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction_moment),reduction_indices=[1]))
#train_step_moment = tf.train.MomentumOptimizer(0.1).minimize(loss_adam)



os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
gpuConfig = tf.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth = True

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
fig.show()

#   windows 环境下运行一定要配置 处理器运行环境 gpu/cpu,否则会报错
#   在windows 下运行本代码可以看到动态变化的效果
with tf.Session(config = gpuConfig) as sess:
    tf.global_variables_initializer().run()
#    with tf.device("/gpu:0"):
    for i in range(2000):
#        sess.run(train_step,feed_dict = {xs:x_data,ys:y_data})
        sess.run(train_step_adam,feed_dict = {xs:x_data,ys:y_data})
        if i % 50 == 0 :
            try:
                ax.lines.remove(lines[0])
#                ax.lines.remove(lines[1])
            except Exception as e:
#                print('error1 :{}'.format(e))
                pass
#            print('trun : {} ; loss : {} '.format(i,sess.run(loss,feed_dict = {xs:x_data,ys:y_data})))
            try:
                prediction_value,prediction_value_adam = sess.run([prediction,prediction_adam],feed_dict = {xs:x_data})
                
#                print('prediction_value :{}'.format(prediction_value))
#                lines = ax.plot(x_data,prediction_value,'r-',lw = 5)
                lines = ax.plot(x_data,prediction_value_adam,'y-',lw = 2)
                plt.pause(0.2)
            except Exception as e:
                print('error2 :{}'.format(e))            
#   
    
    
'''
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(1000):
        sess.run(train_step,feed_dict = {xs:x_data,ys:y_data})
        if i % 50 == 0 :
            prediction_value = sess.run(prediction,feed_dict = {xs:x_data})
            print('trun : {} ; loss : {} '.format(i,sess.run(loss,feed_dict = {xs:x_data,ys:y_data})))
       
#lines = ax.plot(x_data,prediction_value,'r-',lw = 5)
fig.show()
'''

    




