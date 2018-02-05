#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
LR = .1
x = np.linspace(-1,1,200,dtype=np.float32)
REAL_PARAMS = [1.2,2.5]  # 我们假设的需要被学习的真实参数
INIT_PARAMS = [[5,4],[5,1],[2,4.5]][2] # 在调参中不同初始化的参数点
# Test (1): Visualize a simple linear function with two parameters,
# you can change LR to 1 to see the different pattern in gradient descent.

# y_fun = lambda a, b: a * x + b
# tf_y_fun = lambda a, b: a * x + b


# Test (2): Using Tensorflow as a calibrating tool for empirical formula like following.

# y_fun = lambda a, b: a * x**3 + b * x**2
# tf_y_fun = lambda a, b: a * x**3 + b * x**2


# Test (3): Most simplest two parameters and two layers Neural Net, and their local & global minimum,
# you can try different INIT_PARAMS set to visualize the gradient descent.
y_fun = lambda a, b: np.sin(b*np.cos(a*x))
tf_y_fun = lambda a, b: tf.sin(b*tf.cos(a*x))

#####
# lambda the same as 
# def y_fun(a,b):
#     global x
#     return a*x + b

# y_fun 的输入就是不同参数 a b 的值, 和 我们之前定义的数据 x, 输出 y 值. 
# 这里的 tf_y_fun 是传给 tensorflow 去优化的方程.
# y_fun 是用来可视化和计算真是 y 用的.

noise = np.random.randn(200)/10
y = y_fun(*REAL_PARAMS)+noise
#In [26]: print(*y)
# 1 2
# In [27]: print(y)
# [1, 2]

# tensorflow 计算优化图
a,b = [tf.Variable(initial_value=p,dtype=tf.float32) for p in INIT_PARAMS]
pred = tf_y_fun(a,b)
mse = tf.losses.mean_squared_error(y,pred)
# train_op = tf.train.GradientDescentOptimizer(LR).minimize(mse)
train_op = tf.train.AdamOptimizer(0.01).minimize(mse)

a_list,b_list,cost_list = [],[],[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(400):
        a_,b_,mse_ = sess.run([a,b,mse])
        a_list.append(a_);b_list.append(b_);cost_list.append(mse_)
        result,_ = sess.run([pred,train_op])

#visualization
print('a=',a_,'b=',b_)
plt.figure()
plt.scatter(x,y,c='b')
plt.plot(x,result,'r-',lw=2)
# 3D plot
fig = plt.figure(2);ax = Axes3D(fig)
a3D, b3D = np.meshgrid(np.linspace(-2,7,30),np.linspace(-2,7,30)) #parameter space
cost3D = np.array([np.mean(np.square(y_fun(a_,b_)-y)) for a_,b_ in zip(a3D.flatten(),b3D.flatten())]).reshape(a3D.shape)
ax.plot_surface(a3D,b3D,cost3D,rstride=1,cstride=1,cmap='rainbow',alpha=0.5)
ax.scatter(a_list[0],b_list[0],zs=cost_list[0],s=300,c='r')  # initial parameter place
ax.set_xlabel('a'); ax.set_ylabel('b')
ax.plot(a_list, b_list, zs=cost_list, zdir='z', c='r', lw=3)    # plot 3D gradient descent
plt.show()