# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:57:27 2019

@author: Scarlett
基于支持向量(support vector)，scikit-learn主要是包含s三大方面：
1 分类(Classification，SVC、NuSVC、LinearSVC)
2 回归(Regression，SVR、NuSVR、LinearSVR)、
3 异常检测(Outliers detection)

链接：https://www.jianshu.com/p/9e824fa1d421

SVM简介及sklearn参数
https://www.cnblogs.com/solong1989/p/9620170.html


NuSVC　　　　　　　　　　　　　　　　　
class sklearn.svm.NuSVC(nu=0.5, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
 shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None,
 verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None) 　
nu： 训练误差部分的上限和支持向量部分的下限，取值在（0，1）之间，默认是0.5
kernel: 算法中采用的和函数类型，核函数是用来将非线性问题转化为线性问题的一种方法。
        参数选择有RBF, Linear, Poly, Sigmoid，precomputed或者自定义一个核函数,
        默认的是"RBF"，即径向基核，也就是高斯核函数；而Linear指的是线性核函数，Poly指的是多项式核，
        Sigmoid指的是双曲正切函数tanh核；
        
gamma: 核函数系数，该参数是rbf，poly和sigmoid的内核系数；默认是'auto'，那么将会使用特征位数的倒数，
        即1 / n_features。（即核函数的带宽，超圆的半径）。gamma越大，σ越小，使得高斯分布又高又瘦，
        造成模型只能作用于支持向量附近，可能导致过拟合；反之，gamma越小，σ越大，高斯分布会过于平滑，
        在训练集上分类效果不佳，可能导致欠拟合。


"""


print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm



xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                     np.linspace(-3, 3, 500))
np.random.seed(0)
X = np.random.randn(300, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)#   取两列条件一个True 一个False 时的判定结果

# fit the model
clf = svm.NuSVC(gamma='auto')
clf.fit(X, Y)

# plot the decision function for each datapoint on the grid
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linestyles='dashed')
plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
            edgecolors='k')
plt.xticks(())
plt.yticks(())
plt.axis([-3, 3, -3, 3])
plt.show()


# =============================================================================
#   函数说明
# =============================================================================
#   numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None) 在指定的间隔内返回均匀间隔的数字。
#   numpy.meshgrid()    从坐标向量中返回坐标矩阵

# 坐标向量
a = np.array([1,2,3])
# 坐标向量
b = np.array([7,8])
# 从坐标向量中返回坐标矩阵
# 返回list,有两个元素,第一个元素是X轴的取值,第二个元素是Y轴的取值
res = np.meshgrid(a,b)
#返回结果: [array([ [1,2,3] [1,2,3] ]), array([ [7,7,7] [8,8,8] ])] 
#返回两个array 组合结果
b = np.array([7,8,9])
res = np.meshgrid(a,b)

#np.random.randn(d1,d2)
#生成d1*d2 的矩阵 数值符合N~(0,1) 正态分布