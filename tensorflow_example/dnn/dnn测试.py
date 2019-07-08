# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:36:21 2019

@author: Scarlett

https://blog.csdn.net/qq_30666517/article/details/80238729
"""


# construct simple DNN
import numpy as np
import matplotlib.pyplot as plt
import sys
from common import logger
 


def generate_data():
    x = np.linspace(-2,2,100)[np.newaxis,:]
    noise = np.random.normal(0.0,0.5,size=(1,100))
    y = x**2+noise
    logger.debug('x.shape : {} \n x[0] : {}'.format(x.shape,x[0]))
    logger.debug('y.shape : {} \n y[0] : {}'.format(y.shape,y[0]))
    return x,y
 
 
class DNN():
    
    def __init__(self,input_nodes=1,hidden1_nodes=4,hidden2_nodes=4,output_nodes=1):
        self.input_nodes = input_nodes
        self.hidden1_nodes = hidden1_nodes
        self.hidden2_nodes = hidden2_nodes
        self.output_nodes = output_nodes
        self.build_DNN()
        
    def build_DNN(self):
        np.random.seed(1)
        # Layer1 parameter
        self.w1 = np.random.normal(0.0,0.1,size=(self.hidden1_nodes,self.input_nodes))
        self.b1 = np.zeros(shape=(self.hidden1_nodes,1))
        # Layer2 parameter
        self.w2 = np.random.normal(0.0,0.2,size=(self.hidden2_nodes,self.hidden1_nodes))
        self.b2 = np.ones(shape=(self.hidden2_nodes,1))
        # Layer3 parameter
        self.w3 = np.random.normal(0.0,0.5,size=(self.output_nodes,self.hidden2_nodes))
        self.b3 = np.zeros(shape=(self.output_nodes,1))
        
        
        
        
    def forwardPropagation(self,inputs,step):
        if step < 20:
            logger.debug('------- forwardPropagation start ------ ')
            logger.debug('before : \n z1 : {} \n a1 : {} \n z2 : {} \n a2 : {} \n z3 : {} \n a3 : {}'.format(self.z1,self.a1,self.z2,self.a2,self.z3,self.a3))
        self.z1 = np.matmul(self.w1,inputs) + self.b1
        self.a1 = 1/(1+np.exp(-self.z1))
        self.z2 = np.matmul(self.w2,self.a1) + self.b2
        self.a2 = 1/(1+np.exp(-self.z2))
        self.z3 = np.matmul(self.w3,self.a2) + self.b3
        self.a3 = self.z3
        if step < 20:
            logger.debug('before : \n z1 : {} \n a1 : {} \n z2 : {} \n a2 : {} \n z3 : {} \n a3 : {}'.format(self.z1,self.a1,self.z2,self.a2,self.z3,self.a3))        
            logger.debug('------- forwardPropagation end ------ ')
    



    def backwardPropagation(self,da,a,a_1,w,b,last=False):
        '''
        da:current layer activation output partial devirate result
        a:current layer activation output
        a_1:previous layer of current layer activation output
        w:current parameter
        b:current bias
        '''
        # dz = da/dz
        if last:
            dz = da
        else:
            dz = a*(1-a)*da
        # dw = dz/dw
        nums = da.shape[1]
        dw = np.matmul(dz,a_1.T)/nums
        db = np.mean(dz,axis=1,keepdims=True)
        # da_1 = dz/da_1
        da_1 = np.matmul(w.T,dz)
        
        w -= 0.5*dw
        b -= 0.5*db
        return da_1
    
    def train(self,x,y,max_iter=50000):
        for i in range(max_iter):
            self.forwardPropagation(x,i)
            #print(self.a3)
            loss = 0.5*np.mean((self.a3-y)**2)
            if step < 20:
                logger.debug('step : {} ------ '.format(i))
                logger.debug('loss : {} ------ '.format(loss))
            
            
            da = self.a3-y
            da_2 = self.backwardPropagation(da,self.a3,self.a2,self.w3,self.b3,True)
            da_1 = self.backwardPropagation(da_2,self.a2,self.a1,self.w2,self.b2)
            da_0 = self.backwardPropagation(da_1,self.a1,x,self.w1,self.b1)
            self.view_bar(i+1,max_iter,loss)
        return self.a3
    
    def view_bar(self,step,total,loss):
        rate = step/total
        rate_num = int(rate*40)
        r = '\rstep-%d loss value-%.4f[%s%s]\t%d%% %d/%d'%(step,loss,'>'*rate_num,'-'*(40-rate_num),
                                      int(rate*100),step,total)
        sys.stdout.write(r)
        sys.stdout.flush()
        
if __name__ == '__main__':
    x,y = generate_data()
    plt.scatter(x,y,c='r')
    plt.ion()
    print('plot') 
    dnn = DNN()
    predict = dnn.train(x,y)
    plt.plot(x.flatten(),predict.flatten(),'-')
    plt.show()
    
 '''
from datetime import datetime as dt
import sys
import time


for i in range(20):
    a =dt.now()
    sys.stdout.write("\r{0}".format(a))
    sys.stdout.flush()
#    sys.stdout.write("\033[4A")
    time.sleep(1)
    
    
for i in range(100):
    percent = i/100
    sys.stdout.write("\r{0}{1}".format("|"*i, '%.2f%%'%(percent*100)))
    sys.stdout.flush()
    time.sleep(0.5)
        
    
    
      