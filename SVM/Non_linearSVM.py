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
#   NuSVC 通过参数控制支持向量数量
clf = svm.NuSVC(gamma='auto')
#   继承BaseLibSVM 中的fit
clf.fit(X, Y)

# plot the decision function for each datapoint on the grid
#   np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
#   继承BaseSVC 中的decision_function，计算样本x到分离超平面的距离
#   xx.ravel() 将array 一行行追加，降维：xx[:2].shape =(2,500);len(xx[:2].ravel()) = 1000
#   np.c_[xx.ravel(), yy.ravel()] .shape = (250000,2)
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#   array.reshape 将一维重新恢复原来的行列分布
Z = Z.reshape(xx.shape)

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linestyles='dashed')
plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
            edgecolors='k')
# 设置横轴记号为空
#plt.xticks(())
## 设置纵轴记号为空
#plt.yticks(())
#axis([x_left, x_right, y_bottom, y_top])是用来设置所绘制图形的视窗大小的
plt.axis([-3, 3, -3, 3])
plt.show()

'''
热图绘制
https://matplotlib.org/2.2.3/api/_as_gen/matplotlib.pyplot.imshow.html
imshow(X, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None,
           vmin=None, vmax=None, origin=None, extent=None, shape=None,
           filternorm=1, filterrad=4.0, imlim=None, resample=None, url=None,
           hold=None, data=None, **kwargs):
    
色彩参数
plt.cm.summer
    https://matplotlib.org/gallery/color/colormap_reference.html
    
色块效果参数
interpolation
https://blog.csdn.net/liangjiubujiu/article/details/80420555

extent 横纵坐标范围 X_L ,X_H,Y_L,Y_H

ascept :{'equal'|'auto'}如果是“自动”，则更改图像宽高比以匹配轴的宽高比。

origin:
    将数组的[0,0]索引放置在轴的左上角或左下角。约定“上”通常用于矩阵和图像。如果未给出，
    则使用rcparams[“image.origin”]，默认为“upper”。
    请注意，垂直轴向上指向“下部”，向下指向“上部”。
    
plt.contour
https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contour.html
'''
from matplotlib import pyplot as plt
 
X = [[1,2],[3,4],[5,6],[8,10]]

plt.subplot(2,3,1)
plt.imshow(X)
plt.colorbar(cax = None,ax=None,shrink=0.5)

plt.subplot(2,3,2)
plt.imshow(X,cmap = plt.cm.summer)
plt.colorbar(shrink=0.5)

plt.subplot(2,3,3)
plt.imshow(X,cmap = plt.cm.Pastel1)
plt.colorbar()

plt.subplot(2,3,4)
plt.imshow(X,cmap = plt.cm.autumn)
plt.colorbar(shrink = 0.5,ticks = [-1,0,1])

plt.show() 


X = [[0, 0.25], [0.5, 0.75]]   
 
 
fig = plt.figure()
ax = fig.add_subplot(121)
im = ax.imshow(X, cmap=plt.get_cmap('hot'),interpolation = 'hamming')
plt.colorbar(im, shrink=0.5)
 
ax = fig.add_subplot(122)
im = ax.imshow(X, cmap=plt.get_cmap('hot'), interpolation='spline36',
               vmin=0, vmax=1) 
plt.colorbar(im, shrink=0.2)
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

# =============================================================================
# 
# =============================================================================
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA


def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)


def plot_subfigure(X, Y, subplot, title, transform):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=2).fit(X, Y).transform(X)
    else:
        raise ValueError

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    #   classif 
    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(X, Y)

    #   subplot 创建 2* 2 的子图
    plt.subplot(2, 2, subplot)
    plt.title(title)

    zero_class = np.where(Y[:, 0])
    one_class = np.where(Y[:, 1])
    plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
    plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b',
                facecolors='none', linewidths=2, label='Class 1')
    plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
                facecolors='none', linewidths=2, label='Class 2')

    plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
                    'Boundary\nfor class 1')
    plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
                    'Boundary\nfor class 2')
    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    if subplot == 2:
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.legend(loc="upper left")


plt.figure(figsize=(8, 6))

X, Y = make_multilabel_classification(n_classes=2, n_labels=1,
                                      allow_unlabeled=True,
                                      random_state=1)

plot_subfigure(X, Y, 1, "With unlabeled samples + CCA", "cca")
plot_subfigure(X, Y, 2, "With unlabeled samples + PCA", "pca")

X, Y = make_multilabel_classification(n_classes=2, n_labels=1,
                                      allow_unlabeled=False,
                                      random_state=1)

plot_subfigure(X, Y, 3, "Without unlabeled samples + CCA", "cca")
plot_subfigure(X, Y, 4, "Without unlabeled samples + PCA", "pca")

plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
plt.show()