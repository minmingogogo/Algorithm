# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 10:34:09 2019

@author: Scarlett


原文：https://blog.csdn.net/qq_32864683/article/details/80368135 

"""

import numpy as np
import copy
from sklearn.datasets import load_boston#导入博士顿房价数据集


class LinerRegression:
    M_x = []  #
    M_y = []  #
    M_theta = []  # 参数向量
    trained = False

    def __init__(self):
        pass

    def regression(self, data, target):
        self.M_x = np.mat(data)
        # 每个向量添加一个分量1，用来对应系数θ0
        fenliang = np.ones((len(data), 1))
        print('M_x.shape : {} ; fenliang.shape : {} '.format(self.M_x.shape,fenliang.shape))
        self.M_x = np.hstack((self.M_x, fenliang))
        print('fenliang.shape : {} '.format(fenliang.shape))
        self.M_y = np.mat(target)
        M_x_T = self.M_x.T  # 计算X矩阵的转置矩阵
        self.M_theta = (M_x_T * self.M_x).I * M_x_T * self.M_y.T# 通过最小二乘法计算出参数向量
        print('M_x_T.shape : {} ; M_x.shape : {} ; M_y.T.shape : {} /n M_theta.shape : {}'.format(M_x_T.shape,self.M_x.shape,self.M_y.T.shape,self.M_theta.shape))

        self.trained = True

    def predict(self, vec):
        if not self.trained:
            print("You haven't finished the regression!")
            return
        M_vec = np.mat(vec)
        print('M_vec.shape:{} \n M_vec[0] : {}'.format(M_vec.shape,M_vec[0]))
        fenliang = np.ones((len(vec), 1))
        M_vec = np.hstack((M_vec, fenliang))
        print('M_vec.shape:{} \n M_vec[0] : {}'.format(M_vec.shape,M_vec[0]))
        estimate = np.matmul(M_vec,self.M_theta)
        print('estimate.shape:{}'.format(estimate.shape))
        return estimate


if __name__ == '__main__':
    # 从sklearn的数据集中获取相关向量数据集data和房价数据集target
    data, target = load_boston(return_X_y=True)
    print('data : {} \n data.shape : {} \n target : {} \n target.shape : {}'.format(data[0],data.shape,target[0],target.shape))
    lr = LinerRegression()
    lr.regression(data, target)
    # 提取一批样例观察一下拟合效果#每51步抽一个样本
    test = data[::51]
    M_test = np.mat(test)
    print('M_test.shape : {}'.format(M_test.shape))
    
    #   lis[start:end:step]
    real = target[::51]#实际数值real
    print('real.shape : {}'.format(real.shape))
    estimate=np.array(lr.predict(M_test))#回归预测数值estimate
    #打印结果
    for i in range(len(test)):
        print("实际值:",real[i]," 估计值:",estimate[i,0])
