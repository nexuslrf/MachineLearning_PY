# -*- coding: utf-8 -*-
import tensorflow as tf
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

x_data = np.linspace(-1,1,300).reshape((300,1))
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)+noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction)
	,reduction_indices=[1]))
# reduction_indices=[1] 按行求和
# reduction_indices=[0] 按列求和
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# plot the real data
fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
plt.scatter(x_data, y_data)
plt.ion()#本次运行请注释，全局运行不要注释
plt.show()

for i in range(5000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 20 == 0:
        # to visualize the result and improvement
        try:
            lines[0].remove()
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = plt.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)
plt.ioff()
plt.show()

