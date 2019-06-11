# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
https://www.w3cschool.cn/tensorflow_python/tensorflow_python-l5x72feg.html

"""

import tensorflow as tf
import numpy as np

# =============================================================================
#   一 、tf 系列
# =============================================================================
1 tf.placeholder(dtype, shape=None, name=None)
    dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
    shape：数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
    name：名称


2 tf.add() 将参数相加 
    example:
    a = tf.add(2,5)
    b = tf.multiply(a,3)
    with tf.Session() as sess:
        print(sess.run(b))


3 tf.random_normal()   函数用于从服从指定正太分布的数值中取出指定个数的值。 
    #    https://blog.csdn.net/dcrmg/article/details/79028043 

    tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
    shape: 输出张量的形状，必选
    mean: 正态分布的均值，默认为0
    stddev: 正态分布的标准差，默认为1.0
    dtype: 输出的类型，默认为tf.float32
    seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
    name: 操作的名称
    
    example:
    w1 = tf.Variable(tf.random_normal([2,3],stddev = 1, seed = 1))
    with tf.Session() as sess:
        # sess.run(tf.initialize_all_variables())  #比较旧一点的初始化变量方法
        sess.run(tf.global_variables_initializer())
        print(w1)
        print(sess.run(w1))
        
    变量w1声明之后并没有被赋值，需要在Session中调用run(tf.global_variables_initializer())方法初始化之后才会被具体赋值。
    tf中张量与常规向量不同的是执行"print w1"输出的是w1的形状和数据类型等属性信息，获取w1的值需要调用sess.run(w1)方法。


4 tf.matmul() 将矩阵a乘以矩阵b，生成a * b    
    tf.matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None) 
    参数: 
    a: 一个类型为 float16, float32, float64, int32, complex64, complex128 且张量秩 > 1 的张量。 
    b: 一个类型跟张量a相同的张量。 
    transpose_a: 如果为真, a则在进行乘法计算前进行转置。 
    transpose_b: 如果为真, b则在进行乘法计算前进行转置。 
    adjoint_a: 如果为真, a则在进行乘法计算前进行共轭和转置。 
    adjoint_b: 如果为真, b则在进行乘法计算前进行共轭和转置。 
    a_is_sparse: 如果为真, a会被处理为稀疏矩阵。 
    b_is_sparse: 如果为真, b会被处理为稀疏矩阵。 
    name: 操作的名字（可选参数） 


5 tf.multiply()两个矩阵中对应元素各自相乘

    example
    #   两个矩阵对应元素相乘
    x = tf.constant([[1,2,3],[1,2,3],[1,2,3]])
    y = tf.constant([[0,0,1],[0,0,1],[0,0,1]])
    z = tf.multiply(x,y)
    
    #   两个数相乘
    x1 = tf.constant(1)
    y1 = tf.constant(2)
    z1 = tf.multiply(x1,y1)
    
    #   数和矩阵相乘
    x2 = tf.constant([[1,2,3],[1,2,3],[1,2,3]])
    y2 = tf.constant(2)
    z2 = tf.multiply(x2,y2)
    
    #   两个矩阵相乘  这是矩阵乘积,而不是元素的乘积.
    x3 = tf.constant([[1,2,3],[1,2,3],[1,2,3]])
    y3 = tf.constant([[0,0,1],[0,0,1],[0,0,1]])
    z3 = tf.matmul(x3,y3)
    

    with tf.Session() as sess:
        print(sess.run(z))
        print(sess.run(z1))
        print(sess.run(z2))
        print(sess.run(z3))
        

x = tf.placeholder(tf.float32,shape=(None,2))
w1 = tf.Variable(tf.random_normal([2,3],stddev = 1,seed = 1 ))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1,seed = 1))

a = tf.matmul(x , w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y,feed_dict={x:[[0.7,0.5],[0.2,0.3],[0.3,0.4],[0.4,0.5]]}))
#    print(sess.run(w1))
#    print(sess.run(w2))
w1 = tf.Variable(tf.random_normal([2,3],stddev = 1,seed = 1 ))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1,seed = 1))
    
x_ = tf.constant([[0.7,0.5],[0.2,0.3],[0.3,0.4],[0.4,0.5]])
a_ = tf.matmul(x , w1)
y_ = tf.matmul(a, w2)
#   调用变量
#init = tf.initialize_variables()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)        
    print('x : {}'.format(sess.run(x_)))
    print('w1 : {}'.format(sess.run(w1)))
    print('a : {}'.format(sess.run(a_)))
    print('w2 : {}'.format(sess.run(w2)))
    print('y : {}'.format(sess.run(y_)))


6 tf.global_variables_initializer
    在TensorFlow的世界里，变量的定义和初始化是分开的，所有关于图变量的赋值和计算都要
    通过tf.Session的run来进行。想要将所有图变量进行集体初始化时应该使用
    tf.global_variables_initializer。

7 tf.assign()
    assign (ref ,value ,validate_shape = None ,use_locking = None ,name = None)
    通过将 "value" 赋给 "ref" 来更新 "ref"


8 tf.train.Saver
#    原文：https://blog.csdn.net/index20001/article/details/74322198 

    训练网络后想保存训练好的模型，以及在程序中读取以保存的训练好的模型。
    首先，保存和恢复都需要实例化一个 tf.train.Saver。
    saver = tf.train.Saver()
    然后，训练循环中，定期调用 saver.save() 方法，向文件夹中写入包含当前模型中所有可训练变量的 checkpoint 文件。   
    saver.save(sess, FLAGS.train_dir, global_step=step)
    之后，就可以使用 saver.restore() 方法，重载模型的参数，继续训练或用于测试数据。    
    saver.restore(sess, FLAGS.train_dir)
 
    
9 tf.reduce_sum 求和
    x=np.array([[1, 1, 1],[1, 1, 1]])
    
    
    
    #按列求和
    
    tf.reduce_sum(x, 0) ==> [2, 2, 2]
    
    with tf.Session() as sess:
        print(sess.run(tf.reduce_sum(x, 0)))
    
    #按行求和
    
    tf.reduce_sum(x, 1) ==> [3, 3]
    tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
    
    #行列求和
    
    tf.reduce_sum(x, [0, 1]) ==> 6
    
    tf.reduce_sum(x) ==> 6
10 tf.random_uniform
    tf.random_uniform((4, 4), minval=low,maxval=high,dtype=tf.float32)))返回4*4的矩阵，
    产生于low和high之间，产生的值是均匀分布的
    a = tf.random_uniform([10, 20], -1.0, 1.0)
    with tf.Session() as sess:
        print(sess.run(a))

11 tf.nn.embedding_lookup        
    tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。
    tf.nn.embedding_lookup（params, ids）:
        params可以是张量也可以是数组等，id就是对应的索引。
        
    p = tf.Variable(tf.random_normal([10,1]))
    b = tf.nn.embedding_lookup(p , [1,3])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(b))
        print('-'*20)
        print(sess.run(p))

12 tf.truncated_normal
    tf.truncated_normal(shape, mean, stddev) :
    shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，均值和标准差自己设定。
    这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。
    和一般的正太分布的产生随机数据比起来，这个函数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。

    c = tf.truncated_normal(shape = [10,10],mean = 0 ,stddev = 1)
    with tf.Session() as sess:
        print(sess.run(c))
        
# =============================================================================
#   二 、graph 系列
    图与会话：用计算图来构建网络，用会话来具体执行网络
    http://looop.cn/?p=3365http://looop.cn/?p=3365
    tensorflow之图和会话翻译   
    https://blog.csdn.net/yinkun6514/article/details/79527702    
        
     
# =============================================================================
1 feed_dict
    sess = tf.Session()    
    replace_dict = {a:15}
    print(sess.run(b,feed_dict = replace_dict))
        
2 tf.Session() 与 tf.InteractiveSession()
    #   https://www.cnblogs.com/cvtoEyes/p/9035047.html
    tf.InteractiveSession() 相当于
                                    sess = tf.Sess()
                                    with sess.as_default()
    

3 with tf.name_scope 
    #   https://blog.csdn.net/gg_18826075157/article/details/78368924
    一个深度学习模型的参数变量往往是成千上万的，不加上命名空间加以分组整理，将会成为可怕的灾难。
    TensorFlow的命名空间分为两种，tf.variable_scope和tf.name_scope。
    tf.name_scope(name,default_name=None,values=None)
    
    with tf.name_scope(None,"test_default_name",None):
        op = tf.constant(0)
    print(op.name)
    
    with tf.name_scope("test_name","test_default_name",None):
        op = tf.constant(0)
    print(op.name)
    
    with tf.name_scope("test_name","default_name",None):
        op = tf.constant(0)
    print(op.name)
        
    with tf.name_scope("test_name",None,None):
        op = tf.constant(0)
    print(op.name)
    
    
    
    graph_tensor = tf.Graph()
    with graph_tensor.as_default():
        A = tf.constant(1)
        
    graph_1 = tf.Graph()
    graph_1.as_default()
    with tf.name_scope(None,"namescope_1"):
        op1 = tf.constant(0)
        
    graph_2 = tf.Graph()
    graph_2.as_default()
    with tf.name_scope(None,"namescope_2",[A]):
        op2 = tf.constant(0)
    print(graph_tensor)    
    print(op1.graph == graph_tensor) 
    print(op2.graph == graph_ensor)
    #   通过value 将 tensor 放进 tensor A所在的图中
    
4 tf.variable_scope和tf.name_scope的用法    
#    https://www.jianshu.com/p/3d2ff00edcef
    tf.variable_scope可以让不同命名空间中的变量取相同的名字，
        无论tf.get_variable或者tf.Variable生成的变量
    tf.name_scope具有类似的功能，但只限于tf.Variable生成的变量
    
    with tf.variable_scope('v1'):
        a1 = tf.get_variable(name = 'a1',shape = [1],initializer = tf.constant_initializer(1))
        a2 = tf.Variable(tf.random_normal(shape = [2,3],mean = 0 , stddev = 1),name = 'a2')
  
    with tf.variable_scope('v2'):
        a3 = tf.get_variable(name = 'a1',shape = [1],initializer = tf.constant_initializer(1))
        a4 = tf.Variable(tf.random_normal(shape = [2,3],mean = 0 ,stddev = 1),name = 'a2')
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('a1 : {} \n a2 : {} \n a3 : {} \n a4 : {} '.format(a1.name,a2.name,a3.name,a4.name))
    
    
    
    with tf.name_scope('v3'):
#        a1 = tf.get_variable(name = 'a1',shape = [1],initializer = tf.constant_initializer(1))
        a2 = tf.Variable(tf.random_normal(shape = [2,3],mean = 0 , stddev = 1),name = 'a2')

    with tf.variable_scope('v4'):
#        a3 = tf.get_variable(name = 'a1',shape = [1],initializer = tf.constant_initializer(1))
        a4 = tf.Variable(tf.random_normal(shape = [2,3],mean = 0 ,stddev = 1),name = 'a2')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
#        print('a1 : {} \n a2 : {} \n a3 : {} \n a4 : {} '.format(a1.name,a2.name,a3.name,a4.name))
        print('a2 : {} \n \n a4 : {} '.format(a2.name,a4.name))

    #   variable_scope 会影响 tf.get_variable 与 tf.variable 但是
    #   name_scope 会影响tf.variable 不影响 tf.get_variable

    with tf.variable_scope("name_root"):
        a1 = tf.Variable([1],name = 'var1')
        a2 = tf.get_variable("var2",[1])
    print(a1.name,a2.name)
    
    
    with tf.name_scope("name_root2"):
        a3 = tf.Variable([2],name = 'var1_1')
        a4 = tf.get_variable("var2",[2])
    print(a3.name,a4.name)

    #   在执行一次 a4 会报错，因为不get_variable 不受name_scope 约束 则会存在两个var2 
    with tf.name_scope("name_root2"):
        a3 = tf.Variable([2],name = 'var1_1')
        a4 = tf.get_variable("var2",[2])
    print(a3.name,a4.name)

import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess = tf.InteractiveSession()
with tf.variable_scope("scope1"):
    w1 = tf.get_variable("w1", initializer=4.)
    w2 = tf.Variable(0.0, name="w2")
with tf.variable_scope("scope2"):
    w1_p = tf.get_variable("w1", initializer=5.)
    w2_p = tf.Variable(1.0, name="w2")
with tf.variable_scope("scope1", reuse=True):
    w1_reuse = tf.get_variable("w1")
    w2_reuse = tf.Variable(1.0, name="w2")
with tf.variable_scope('scope2',reuse = True):
    w1_reuse_1 = tf.get_variable("w1",initializer = 555.0)
    w2_reuse_1 = tf.Variable(1.0,name = 'w2')


def compare_var(var1, var2):
    print ('-----------------')
    print('var1 : {} ; \n var2 : {}'.format(var1,var2))
    if var1 is var2:
        print (sess.run(var2))
    print (var1.name, var2.name)
sess.run(tf.global_variables_initializer())

compare_var(w1, w1_p)
compare_var(w2, w2_p)
compare_var(w1, w1_reuse)
compare_var(w2, w2_reuse)    
compare_var(w1, w1_reuse_1)
compare_var(w2_p, w2_reuse_1)


import tensorflow as tf 
 
def fun():
 
    with tf.variable_scope("scope1", reuse = tf.AUTO_REUSE ):
        var = tf.get_variable( name='var1' ,  initializer=9.0 )
    
    with tf.Session() as sess:
        sess.run( tf.global_variables_initializer())
        ar = sess.run(var)
        print('ar : {} '.format(ar))
        with tf.variable_scope("scope1", reuse=tf.AUTO_REUSE):
            var = tf.get_variable(name='var1', shape=[] , dtype = tf.float32 )#, initializer = 3 )
        varP = tf.assign( var,  3.0 )
        #sess.run( var.initializer)
        print('varP : {}'.format( sess.run(varP) ))
        print('varP.name : {}'.format(varP.name))
    for var in tf.global_variables() :
        print( 'var.name : {}'.format(var.name ))
        
fun()







5 tf.InteractiveSession()与tf.Session()    
    意思就是在我们使用tf.InteractiveSession()来构建会话的时候，我们可以先构建一个session
    然后再定义操作（operation），如果我们使用tf.Session()来构建会话我们需要在会话构建之前
    定义好全部的操作（operation）然后再构建会话。    
    
    
   
    
# =============================================================================
#   三 、可视化 tensorBoard
# =============================================================================
https://blog.csdn.net/qq_34018578/article/details/82747905










