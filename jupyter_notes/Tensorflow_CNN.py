import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
#载入 MNIST 的手写数字库
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1}) 
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1) #切断正态分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    #strides format:[1,x_movement,y_movement,1]
    #must have strides[0]=strides[3]=1
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
# 第一个参数x的shape为[batch，height，width，channels]

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#ksize  池化窗口的大小  是一个四位向量  
#一般为[1，height，width，1] 因为我们不想在batch和channels上做操作，所以这两个维度上设为1
# 第三个参数，和卷积类似，窗口在每一个维度上滑动的步长，所以一般设为【1,stride，stride，1】

xs = tf.placeholder(tf.float32,[None,784]) #28x28 pixels
ys = tf.placeholder(tf.float32,[None,10]) #10 slots to judge which number
keep_prob=tf.placeholder(tf.float32)


x_image = tf.reshape(xs,[-1,28,28,1])
#-1的含义是：我们自己不用去管这个维度的大小，
# reshape会自动计算，但是我的这个列表中只能有一个  -1  。
# 原因很简单，多个 -1  会造成多解的方程情况
## conv1 layer##
W_conv1 = weight_variable([5,5,1,32]) #patch 5*5 in size=1(单通道图片),out size=32(输出是32个featuremap)
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) # output size:28x28x32
h_pool1 = max_pool_2x2(h_conv1) #output size 14x14x32

## conv2 layer##
W_conv2 = weight_variable([5,5,32,64]) #patch 5*5 in size=1(单通道图片),out size=32(输出是32个featuremap)
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) # output size:14x14x32
h_pool2 = max_pool_2x2(h_conv2) #output size 7x7x32

## func1 layer ##
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
#[n_samples,7,7,64] --> [n_samples,7*7*64]
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64]) 
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

## func2 layer ##
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction)
                            ,reduction_indices=[1])) #loss 交叉熵
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i%50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
