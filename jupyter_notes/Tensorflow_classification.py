# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data #MNIST  手写数字库
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1) 
    #在机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1。
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
#从 MINIST_data 文件夹载入数据


xs = tf.placeholder(tf.float32,[None,784]) #28x28 pixels
ys = tf.placeholder(tf.float32,[None,10]) #10 slots to judge which number

prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction)
	,reduction_indices=[1])) #loss 交叉熵
# reduction_indices=[1] 按行求和
# reduction_indices=[0] 按列求和
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs}) 
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result


for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    # training
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))


