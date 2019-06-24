# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:45:15 2019

@author: Scarlett
"""



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


def add_layer(inputs,in_size,out_size,n_layer,activation_func = None):
    layer_name = 'layer%s'%n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name = 'W')
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+ 0.1 ,name = 'b')
            tf.summary.histogram(layer_name+'/biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs,Weights), biases)
        if activation_func is None:
            outputs  = Wx_plus_b
        else:
            outputs = activation_func(Wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs


    
    
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name = 'x_input')
    ys = tf.placeholder(tf.float32,[None,1],name = 'y_input')


l1 = add_layer(xs,1,10,n_layer ='1',activation_func=tf.nn.relu)
prediction = add_layer(l1,10,1,n_layer ='2',activation_func=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

merged = tf.summary.merge_all()
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
    writer = tf.summary.FileWriter('logs',sess.graph)
#    with tf.device("/gpu:0"):
    for i in range(2000):
        sess.run(train_step,feed_dict = {xs:x_data,ys:y_data})
        if i % 50 == 0 :
            result = sess.run(merged,feed_dict = {xs:x_data,ys:y_data})
            writer.add_summary(result,i)
            try:
                ax.lines.remove(lines[0])
            except Exception as e:
                pass
            try:
                prediction_value = sess.run(prediction,feed_dict = {xs:x_data})                
                lines = ax.plot(x_data,prediction_value,'r-',lw = 5)
                plt.pause(0.1)
            except Exception as e:
                print('error2 :{}'.format(e))            
#   
    
# =============================================================================
#    cmd ->     tensorboard --logdir=logs
# =============================================================================
    

    




