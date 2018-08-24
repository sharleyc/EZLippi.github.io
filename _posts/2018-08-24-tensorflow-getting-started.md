---
layout:     post
title:      TensorFlow入门篇
keywords:   博客
categories: [机器学习]
tags:	    [TensorFlow，MNIST]
---



## MNIST机器学习入门

MNIST好比编程入门的Hello World。 MNIST是一个入门级的计算机视觉数据集，它包含各种手写数字图片及其标签：   

   ![](/images/images_2018/MNIST.png)

上面四张图片的标签分别是5，0，4，1。

我们将训练一个机器学习模型来预测图片里面的数字。我们先使用一个简单的数学模型Softmax Regression来实现预测。完整代码如下：    


	import tensorflow as tf
	import tensorflow.examples.tutorials.mnist.input_data as input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	
	x = tf.placeholder(tf.float32,[None, 784])
	W = tf.Variable(tf.zeros([784,10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.nn.softmax(tf.matmul(x,W) + b)
	y_ = tf.placeholder("float",[None,10])
	cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	for i in range(1000):
	    batch_xs,batch_ys = mnist.train.next_batch(100)
	    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
	print(sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

### Softmax回归

Softmax回归是逻辑回归模型在多分类问题上的推广，标签 y 可以取 k 个不同的值，在MNIST数字识别任务中，k = 10（当k = 2时，softmax回归就退化为逻辑回归了）。用向量来表示softmax回归的计算过程如下图：

   ![](/images/images_2018/softmax-regression-vectorequation.png)

写成更紧凑的方式是：   

y = softmax(Wx + b)


### 实现回归模型

#### 占位符和变量的使用

`x = tf.placeholder(tf.float32,[None, 784])`，x是一个占位符placeholder，表示输入任意数量的MNIST图像，每一张图展平成784维的向量，这里用2维的浮点数张量来表示这些图，张量的形状是[None, 784],None表示张量的第一个维度可以是任何长度的。

模型的权重值和偏置量，用Variable表示。一个Variable代表一个可修改的张量。这里用全为零的张量来初始化W和b：`W = tf.Variable(tf.zeros([784,10]))` 和 `b = tf.Variable(tf.zeros([10]))`。

占位符是定义一个**可变的常量**，占位符赋值后不用初始化就可以获取值。变量必须初始化才有具体的值。

#### 模型的实现和训练

模型的实现只用了一行代码：`y = tf.nn.softmax(tf.matmul(x,W) + b)`。为了训练模型，需要定义一个指标来评估模型的好坏，在机器学习中这个指标成为损失。我们需要尽量最小化损失。交叉熵(cross-entropy)是一个非常常见的损失函数：


   ![](/images/images_2018/cross-entropy.png)

为了计算交叉熵，需要添加一个新的占位符y_，用来输入正确值。计算交叉熵的代码为`cross_entropy = -tf.reduce_sum(y_ * tf.log(y))`，注意交叉熵衡量的不是单一样本的损失，而是所有样本的损失。接下来，我们用梯度下降算法以0.01的学习速率最小化交叉熵。在运行计算前，需要初始化我们创建的变量：`init = tf.initialize_all_variables()`。

现在可以在一个Session里启动模型`sess = tf.Session()`，并初始化变量`sess.run(init)`。然后开始训练模型，使用的是小批量梯度下降算法，一次处理100个数据点，循环1000次。

#### 模型的评估

tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，比如`tf.argmax(y,1)`返回的是模型对于任一输入x预测到的标签值，而 `tf.argmax(y_,1)` 代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)：`tf.equal(tf.argmax(y,1), tf.argmax(y_,1))`。

代码会返回一组布尔值，我们把布尔值转换成浮点数，然后取平均值，例如[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75：`tf.reduce_mean(tf.cast(correct_prediction,"float"))`。

最后计算模型在测试数据集上的正确率：`sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels})`

执行结果为：0.9198。这个数据并不好，因为模型很简单。下面将为MNIST构建一个深度卷积神经网络，来获取更高的正确率。








