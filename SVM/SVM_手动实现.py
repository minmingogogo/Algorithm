# -*- coding: utf-8 -*-
"""
Created on Thu May 23 18:42:40 2019

@author: Scarlett

支持向量机的Python语言实现
http://www.cnblogs.com/pursued-deer/p/7892342.html

https://zhuanlan.zhihu.com/p/29604517
"""

#-*- coding=utf-8 -*-
import random

from numpy import *
from common import logger

# 将文本中的样本数据添加到列表中
#    fileName = 'testSetRBF2.txt'
#    a = '-0.214824    0.662756    -1.000000'
#    a.strip().split('   ')
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName, encoding='utf-8')
    for line in fr.readlines() : # 对文本按行遍历
#        lineArr = line.strip().split('\t')
        lineArr = line.strip().split('   ') #   三个空格
        dataMat .append([float(lineArr [0]), float(lineArr[1])])   # 每行前两个是属性数据，最后一个是类标号
        labelMat .append(float(lineArr[2]))
    return dataMat , labelMat

# 随机选取对偶因子alpha ,参数i 是alpha 的下标，m 是alpha 的总数
def selectJrand(i,m):
    j = i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

# 对所求的对偶因子按约束条件的修剪
def clipAlpha(aj, H, L): # H 为上界，L为下界
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn # 样本数据
        self.labelMat = classLabels # 样本的类标号
        self.C = C # 对偶因子的上界值 松弛变量
        self.tol = toler    #   容错率
        self.m = shape(dataMatIn)[0] # 样本的行数，即样本对象的个数
        self.alphas = mat(zeros((self.m, 1))) # 对偶因子
        self.b = 0 # 分割函数的截距
        self.eCache = mat(zeros((self.m, 2))) # 差值矩阵 m * 2，第一列是对象的标志位 1 表示存在不为零的差值 0 表示差值为零，第二列是实际的差值 E
        self.K = mat(zeros((self.m, self.m))) # 对象经过核函数映射之后的值
        for i in range(self.m): # 遍历全部样本集
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup ) # 调用径向基核函数

#   kernelTrans(X, A, kTup): # X 是样本集矩阵，A 是样本对象（矩阵的行向量） ， kTup 元组

# 预测的类标号值与真实值的差值，参数 oS 是类对象，k 是样本的对象的标号
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)  # 公式（1）
    Ek = fXk - float(oS.labelMat[k]) # 差值
    return Ek


# 由启发式选取第二个 alpha，以最大步长为标准
def selectJ(i, oS, Ei): # 函数的参数是选取的第一个 alpha 的对象号、类对象和对象差值
    maxK = -1; maxDeltaE = 0; Ej = 0 # 第二个 alpha 的初始化
    oS.eCache[i] = [1,Ei] # 更新差值矩阵的第 i 行
    validEcacheList = nonzero(oS.eCache[:,0].A)[0] # 取差值矩阵中第一列不为 0 的所有行数（标志位为 1 ），以元组类型返回
    if (len(validEcacheList)) > 1 : #
        for k in validEcacheList : # 遍历所有标志位为 1 的对象的差值
            if k == i: continue
            Ek = calcEk(oS, k) # 计算对象 k 的差值
            deltaE = abs(Ei - Ek) # 取两个差值之差的绝对值
            if (deltaE > maxDeltaE): # 选取最大的绝对值deltaE
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej # 返回选取的第二个 alpha
    else: # 随机选取第二个 alpha
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS,j)
    return j, Ej # 返回选取的第二个 alpha

# 更新差值矩阵的数据
def updateEk(oS, k):
    print('---------------updateEk %s  -------------'%k)
#    print('更新前 oS.eCache [k]')
#    print(oS.eCache [k])
    logger.debug('---------------updateEk  %s  -------------'%k)
    logger.debug('before  k :{}'.format(k))
    logger.debug('oS.eCache [k] : {}'.format(oS.eCache [k]))
        
    Ek = calcEk(oS, k) # 调用计算差值的函数
    oS.eCache [k] = [1,Ek]
#    print('更新后 oS.eCache [k]')
#    print(oS.eCache [k])
    logger.debug('after oS.eCache [k] : {}'.format(oS.eCache [k]))
    logger.debug('--------------updateEk end- -------------')
    
# 优化选取两个 alpha ，并计算截距 b
def innerL(i, oS):
    Ei = calcEk(oS, i) # 计算 对象 i 的差值
    # 第一个 alpha 符合选择条件进入优化
    if ((oS.labelMat [i]*Ei <- oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat [i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        print('---------------innerL %s  -------------'%i)
        logger.debug('---------------innerL %s optimize -------------'%i)
        j,Ej =selectJ(i, oS, Ei) # 选择第二个 alpha
        alphaIold = oS.alphas[i].copy() # 浅拷贝
        alphaJold = oS.alphas[j].copy() # 浅拷贝
        logger.debug("i : {}, alphaIold : {} , j: {} , alphaJold : {}".format(i,alphaIold,j,alphaJold))

        # 根据对象 i 、j 的类标号（相等或不等）确定KKT条件的上界和下界
        if (oS.labelMat[i] != oS.labelMat[j]):
            logger.debug('oS.labelMat[i] != oS.labelMat[j]')
            L = max(0, oS.alphas [j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else :
            logger.debug('oS.labelMat[i] == oS.labelMat[j]')
            L = max(0, oS.alphas[j] + oS.alphas [i] - oS.C)
            H = min(oS.C, oS.alphas [j] + oS.alphas [i])

        logger.debug("L : {}, H : {}".format(L,H))

        if L==H:
            print ("L==H")
            logger.debug("L==H")
            return 0 # 不符合优化条件（第二个 alpha）
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]  # 计算公式的eta,是公式的相反数
        #   a2_new = a2_old +y2(E1-E2)/eta
        logger.debug("eta : {}".format(eta))
        if eta >= 0:
            print ("eta>=0")
            logger.debug("eta>=0")
            return 0 # 不考虑eta 大于等于 0 的情况（这种情况对 alpha 的解是另外一种方式，即临界情况的求解）
        # 优化之后的第二个 alpha 值 a2_New = a2_Old + y2(E1-E2)/eta
#        print('更新前oS.alphas[j] ：',oS.alphas[j])
        logger.debug('before oS.alphas[j] :{}'.format(oS.alphas[j]))
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L) #  
#        print('更新后oS.alphas[j] ：',oS.alphas[j])
        logger.debug('after oS.alphas[j] :{}'.format(oS.alphas[j]))
   
        updateEk(oS, j) # 更新差值矩阵
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): # 优化之后的 alpha 值与之前的值改变量太小，步长不足
            print ("j not moving enough")
            logger.debug('abs(oS.alphas[j] - alphaJold) < 0.00001')
            logger.debug('j not moving enough')
            return 0
        logger.debug('before oS.alphas[i] :{}'.format(oS.alphas[i]))
        #   a1_new = a1_old + y1y2(a2_old - a2_new)
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j]) # 优化第二个 alpha
        logger.debug('agter oS.alphas[i] :{}'.format(oS.alphas[i]))
        updateEk(oS, i) # 更新差值矩阵
        # 计算截距 b
        #   b1_new = b_old -E1 - y1(a1_new-a1_old)x1Tx1-y2(a2_new - a2_old)x2Tx1
        #   b2_new = b_old -E2 - y1(a1_new-a1_old)x1Tx2-y2(a2_new - a2_old)x2Tx2
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        logger.debug('b1 :{} , b2 : {} , oS.b :{}'.format(b1,b2,oS.b))
        if (0 < oS.alphas [i]) and (oS.C > oS.alphas[i]):
            logger.debug('(0 < oS.alphas [i]) and (oS.C > oS.alphas[i]) ,oS.b = b1')
            oS.b = b1
        elif (0 < oS.alphas [j]) and (oS.C > oS.alphas [j]):
            logger.debug('(0 < oS.alphas [j]) and (oS.C > oS.alphas [j]) ,oS.b = b2')
            oS.b = b2
        else :
            logger.debug('oS.b = (b1 + b2)/2.0')
            oS.b = (b1 + b2)/2.0
        return 1 # 进行一次优化
    else :
        logger.debug('---------------innerL %s not any optimize -------------'%i)
        ((oS.labelMat [i]*Ei <- oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat [i]*Ei > oS.tol) and (oS.alphas[i] > 0))
        logger.debug('oS.labelMat [i] :{} ; Ei :{} ; oS.tol : {} ; oS.alphas[i] : {}'.format(oS.labelMat [i],Ei,oS.tol ,oS.alphas[i]))
        return 0


# 遍历所有能优化的 alpha
#   smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
#    k1=1.3
#    dataMatIn = dataArr
#    classLabels = labelArr  
#    C = 200
#    toler = 0.0001
#    maxIter = 10000
#    kTup=('rbf', k1)
    
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    print('---------------smoP -------------')
    logger.debug('---------------smoP -------------')
    
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup) # 创建一个类对象 oS ,类对象 oS 存放所有数据
    #   打印 K
    logger.debug('oS.K :')
    temp_list = oS.K.tolist()
    for i in range(len(temp_list)):
        temp = list(map(lambda x : '%.4f'%x,temp_list[i]))
        logger.debug(" {}".format(temp))
    
    iter = 0 # 迭代次数的初始化
    entireSet = True # 违反 KKT 条件的标志符
    alphaPairsChanged = 0 # 迭代中优化的次数

    # 从选择第一个 alpha 开始，优化所有alpha
    logger.debug(u"---------------开始 alpha 优化------------")
    while(iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet )): # 优化的终止条件：在规定迭代次数下，是否遍历了整个样本或 alpha 是否优化
        alphaPairsChanged  = 0
        if entireSet: 
            logger.debug(u"存在违反 KKT 的alpha  继续迭代")
            for i in range(oS.m): # 遍历所有对象
#                print('alphaPairsChanged update :',i)
                alphaPairsChanged += innerL(i ,oS) # 调用优化函数（不一定优化）
                logger.debug('---------------innerL end ----------------')
            print ("fullSet , iter: %d i %d, pairs changed %d" % (iter, i , alphaPairsChanged ))
            logger.debug("fullSet , iter: %d i %d, pairs changed %d" % (iter, i , alphaPairsChanged ))
            iter += 1 # 迭代次数加 1
        else:
            logger.debug(u"不存在违反 KKT 的计算 遍历 0 <oS.alphas <C 的样本")
            #   oS.alphas .A 是什么？？
            #   Matrix.A : 表示将矩阵转为 array
            nonBoundIs = nonzero((oS.alphas .A > 0) * (oS.alphas.A < C))[0]
            logger.debug('nonBoundIs : {}'.format(nonBoundIs))
            for i in nonBoundIs : # 遍历所有非边界样本集
                alphaPairsChanged += innerL(i, oS) # 调用优化函数（不一定优化）
                logger.debug('---------------innerL end ----------------')
                print ("non-bound, iter: %d i :%d, pairs changed %d" % (iter, i, alphaPairsChanged ))
                logger.debug ("non-bound, iter: %d i :%d, pairs changed %d" % (iter, i, alphaPairsChanged ))
            iter += 1 # 迭代次数加 1
        if entireSet : # 没有违反KKT 条件的alpha ，终止迭代
            entireSet = False
            logger.debug(u"没有违反KKT 条件的alpha ，终止迭代")
        elif (alphaPairsChanged == 0): # 存在违反 KKT 的alpha
            entireSet = True
            logger.debug(u"存在违反 KKT 的alpha  继续迭代")
        
        logger.debug ("iteration number: %d" % iter)
    logger.debug(u"---------------alpha 优化 结束------------")
    logger.debug('oS.b : {}'.format(oS.b))
    logger.debug('oS.alphas : {}'.format(oS.alphas))    
    logger.debug('---------------smoP end---------------')    
    return oS.b, oS.alphas # 返回截距值和 alphas

# 径向基核函数（高斯函数）
def kernelTrans(X, A, kTup): # X 是样本集矩阵，A 是样本对象（矩阵的行向量） ， kTup 元组
    logger.debug('---------------kernelTrans star-------------')
    
    m,n = shape(X)
    K = mat(zeros((m,1)))   # 初始化K 为 0 矩阵
    # 数据不用核函数计算
    if kTup [0] == 'lin':
        logger.debug(u'线性核函数')
        logger.debug('X:')
        logger.debug('{}'.format(X))
        logger.debug('A.T:')
        logger.debug('{}'.format(A.T))

        K = X * A.T     #   更新K = X矩阵乘以X自己的某行
        logger.debug('K = X * A.T:')
        logger.debug('{}'.format(K))
    # 用径向基核函数计算
    elif kTup[0] == 'rbf':
        logger.debug(u'高斯核函数')
        for j in range(m):
            logger.debug('X[j,:] : {},A:{}'.format(X[j,:], A))
            deltaRow = X[j,:] - A   #   每行与第 A 行做差
            K[j] = deltaRow * deltaRow.T    # 差值内积（x-入)^2 
            logger.debug('deltaRow = {};K[j] = {}'.format(deltaRow,K[j]))
        K = exp(K/(-1*kTup[1]**2))    # 高斯核公式，exp(-（X - 入)^2 /θ^2)
        logger.debug('K = exp(K/(-1*kTup[1]**2)):')
        logger.debug('{}'.format(K))        
    # kTup 元组值异常，抛出异常信息
    else:raise NameError('Houston We Have a Problem --That Kernel is not recognized')
#    print('K:')
#    print(K)
    logger.debug('---------------kernelTrans end-------------')
    return K

# 训练样本集的错误率和测试样本集的错误率
def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt') # 训练样本的提取
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) # 计算得到截距和对偶因子alphas
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()    #转置
    svInd = nonzero(alphas.A>0)[0] # 对偶因子大于零的值，支持向量的点对应对偶因子
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print ("there are %d Support Vectors" % shape(sVs)[0])
    logger.debug("there are %d Support Vectors" % shape(sVs)[0])
    
    logger.debug("labelSV.shape : {}".format(labelSV.shape))
    logger.debug("alphas.shape : {}".format(alphas[svInd].shape))

    
    #   如何筛选支持向量？？？alphas > 0 ？
    m,n = shape(datMat)
    errorCount = 0
    # 对训练样本集的测试
    logger.debug(u'-------------训练样本测试----------' )
    logger.debug('multiply(labelSV,alphas[svInd]) :')
    temp = multiply(labelSV,alphas[svInd]).tolist()
    for ind in range(len(temp)):
        logger.debug('{}'.format(temp[ind]))
    
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1)) # 对象 i 的映射值
        logger.debug('kernelEval.shape = {}'.format(kernelEval.shape))
        #   multiply 计算的是相同位置的乘积 是数量积，非矩阵的矢量积
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b # 预测值
        #   sign 函数
        logger.debug('predict = {}; sign(predict) ={}; sign(labelArr[i])={}'.format(predict,sign(predict),sign(labelArr[i])))
        if sign(predict)!=sign(labelArr[i]):
            errorCount += 1
    print ("the training error rate is: %f" % (float(errorCount)/m))
    logger.debug ("the training error rate is: %f" % (float(errorCount)/m))


    dataArr,labelArr = loadDataSet('testSetRBF2.txt') # 测试样本集的提取
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    # 对测试样本集的测试
    logger.debug(u'----------对测试样本集的测试---------' )
    logger.debug('sVs 是训练的数据 dataMat 是测试的数据，multiply(labelSV,alphas[svInd]) 是训练的结果')
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1)) # 测试样本对象 i 的映射值
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b # 预测值
        logger.debug('predict = {}; sign(predict) ={}; sign(labelArr[i])={}'.format(predict,sign(predict),sign(labelArr[i])))
        if sign(predict)!=sign(labelArr[i]): 
            errorCount += 1
    print ("the test error rate is: %f" % (float(errorCount)/m))
    logger.debug ("the test error rate is: %f" % (float(errorCount)/m))
    
if __name__ == '__main__':


    # 显示计算的截距值b 和对偶因子 alphas
    dataMat ,labelMat = loadDataSet('testSet.txt')
    b, alphas = smoP(dataMat, labelMat , 0.6, 0.11, 40,('rbf',2))
    print ('------')
    print (b, '----',alphas)


    # 支持向量机的测试
    testRbf()


