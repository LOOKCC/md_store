---
title: learn_tensorflow-3
date: 2017-12-23 02:53:59
tags: 机器学习
---
# tensorflow学习心得3-实战MNIST
前两篇讲的都比较笼统，只是大致介绍了tensorflow，但并没有具体讲怎么用，这个一次就参照官网上的教程，来使用minst进行实战。
# 数据集介绍
这是一个很简单的手写数字识别的数据集，大小为28*28，我们的任务就是分辨出上面写的是数字几，虽然图像问题使用CNN会比较好，但是因为数据集很简单，所以我们可以使用一个很简单的感知机模型了来完成这个分类任务。CNN将在下次的cifar10中讲解。
# 模型介绍
每一张图片，我们将它化成行向量，就是长度为784的行向量，这样的数据一共有55000张，所以我们的输入就是一个[55000,784]的矩阵（或者说是一个张量)，而对于每张图片，他将有10类(0-9)，是那个数字的位置为一，其他的位置为0，比如2就是[0,0,1,0,0,0,0,0,0,0]，那么，我们的输出就是[55000,10]。我们按照官网上的模型，那么就是output = softmax(W*input + b) 我们的W是一个[784,10]的矩阵，b是一个[10,1]的矩阵（或者说是一个列向量）。
# 开始构建我们的网络
## 导入数据
因为MNIST是一个很出名的数据集，所以tensorflow的官方的包里就有这个，我们直接使用它包里的导入.  
```
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```
这段代码会在你执行的时候去下载相关的数据集  
## 图构建
接下来是相关的变量的定义
```
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_ = tf.placeholder(tf.float32, [None, 10])
```
这里的x和y_ 就是我们的input和output，可以看出，他们和我们的定义的是一样的，只是将那里的55000换成了None，因为我们的训练集的大小是可以改变的。之后的w和b就和我们模型中叙述的一模一样了，要注意w和b是Variable类型的，x和y_是先创建了一个占位符。  
之后是我们的计算方法，也就是Operator
```
y = tf.nn.softmax(tf.matmul(x, W) + b)
```
下来是优化方法
```
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```
先定义损失，然后可选择优化方法，优化刚刚定义的损失.
## 会话
接先来进入最重要的部分，进入我们的图和和会话
```
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```
首先告诉它，我们直接使用用于交互上下文的会话，这里的交互上下文的意思就像python那样，但这是一个用于临时的交互，并不是正式的写法。（下面我将给出一个正式一些的写法。)之后初始化参数，然后迭代训练。
下面是比较正式的写法：
```
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
sess.close()
```
但是，为了防止我们忘记关掉sess，还有一种写法
```
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```
其中最后一种比较常用一些。
## 检测
优化部分已经解决了，那么下面我们要解决的就是检测问题，我们要证明我们训练的结果是对的，也就是说我们要把我们的训练好的模型放在测试集上跑一跑，看看效果。
具体的代码就是这样：
```
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
```
# 总的代码
```
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_ = tf.placeholder(tf.float32, [None, 10])
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
```
