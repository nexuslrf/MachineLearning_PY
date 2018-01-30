# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#  add_layer() 方法中添加一个参数 n_layer,用来标识层数, 并且用变量 layer_name 代表其每层的名名称
# tensorflow中提供了tf.histogram_summary()方法,用来绘制图片, 第一个参数是图表的名称, 第二个参数是图表要记录的变量
def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name='layer%d'%n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='biases') 
            tf.summary.histogram(layer_name+'/biases',biases)
        #在机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1。
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights)+biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs

x_data = np.linspace(-1,1,300).reshape((300,1))
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)+noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')
#add hidden layer
l1 = add_layer(xs,1,10,1,activation_function=tf.nn.relu)
#add output layer
prediction = add_layer(l1,10,1,2,activation_function=None)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction)
	   ,reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
# loss是在tesnorBorad 的event下面的, 这是由于我们使用的是tf.scalar_summary() 方法.
# reduction_indices=[1] 按行求和
# reduction_indices=[0] 按列求和


with tf.name_scope('Train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

merged = tf.summary.merge_all() 
# tf.merge_all_summaries() 方法会对我们所有的 summaries 合并到一起.
init = tf.global_variables_initializer()

sess = tf.Session()
writer = tf.summary.FileWriter("logs/",sess.graph)

sess.run(init)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%20==0:
        result = sess.run(merged,
            feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)

# powershell input :tensorboard --logdir logs
