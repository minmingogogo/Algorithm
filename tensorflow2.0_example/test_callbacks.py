# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:23:22 2019

@author: Scarlett
"""

# =============================================================================
# part 5 回调函数 callbacks
#        earlyStopping：提前停止训练 ,modelCheckpoing:参数中间状态保存,tensorboard
# =============================================================================

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
    
fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(
        x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_valid_scaled = scaler.transform(
        x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_test_scaled = scaler.transform(
        x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28,28]),
        keras.layers.Dense(300, activation = 'relu'),
        keras.layers.Dense(100, activation = 'relu'),
        keras.layers.Dense(10, activation = 'softmax')])
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = 'sgd',
              metrics = ['accuracy'])  

#   在 fit 过程中 使用回调函数  
#   tensorboard 需要文件夹；ModelCheckpoint 需要一个文件名
logdir = './callbacks'
#logdir = 'C:/Users/10651/MyGit/test_tensorflow_2.0/callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)


    
output_model_file = os.path.join(logdir,'fashion_mnist_model.h5')
#   ModelCheckpoint 中如果没有save_best_only 默认保存最近一个模型
#   EarlyStopping 关键参数：monitor: 关注的指标 ；min_delta :阈值，两次训练的差；
#                 patience: 连续多少次发生差值小于 min_delta 就关闭

#   注意tensorflow log_dir 路径写法 keras.callbacks.TensorBoard('./callbacks')会报错

callbacks = [
            keras.callbacks.TensorBoard('callbacks'),
            keras.callbacks.ModelCheckpoint(output_model_file,
                                            save_best_only = True),
            keras.callbacks.EarlyStopping(patience = 5 , min_delta = 1e-5)
            ]
try:
    history = model.fit(x_train_scaled,y_train,epochs = 10,
              validation_data = (x_valid_scaled,y_valid),
              callbacks = callbacks)
except Exception as e:
    print(e)
    
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)# 显示网格
    plt.gca().set_ylim(0,1)#    设置坐标轴范围
    plt.show()
    
plot_learning_curves(history)

#   测试集测试
#model.evaluate(x_test,y_test)
# loss: 33.7887 - accuracy: 0.6986
model.evaluate(x_test_scaled,y_test)
# loss: 0.4231 - accuracy: 0.8495
#归一化后再测试集上表现明显提升
