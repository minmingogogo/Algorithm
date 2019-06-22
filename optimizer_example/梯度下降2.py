# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 18:19:08 2019

@author: Scarlett

https://blog.csdn.net/titan0427/article/details/50365480

"""

#x = [100,80,120,75,60,43,140,132,63,55,74,44,88]
#y = [120,92,143,87,60,50,167,147,80,60,90,57,99]
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
# y = mx + b
# m is slope, b is y-intercept
#求误差函数，就是预测值和实际值相减然后求平方，最后再取平均值
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    #将数据分别赋值为x,y
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

#梯度下降算法实现，迭代一次后输出一组参数值
def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]



def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]


def run():
    points = genfromtxt('/Users/cailei/Cai_Lei/AI/Testdata/GradientDescentData1.csv', delimiter=",")
    data = pd.read_csv('/Users/cailei/Cai_Lei/AI/Testdata/GradientDescentData1.csv',names=['x','y'])
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    print "Running..."
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))

    plt.plot(data.x,data.y,'bo')
    plt.plot(data.x,data.x*m+b)
    plt.show()


if __name__ == '__main__':
    run()
