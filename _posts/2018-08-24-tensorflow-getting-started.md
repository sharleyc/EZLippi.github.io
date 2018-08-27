---
layout:     post
title:      TensorFlow入门篇（一）
keywords:   博客
categories: [机器学习]
tags:	    [TensorFlow，MNIST]
---

## TensorFlow基本概念

TensorFlow的名字中已经说明了它最重要的两个概念——Tensor和Flow。Tensor就是张量。在TensorFlow中，所有的数据都通过张量的形式来表示。从功能的角度上看，张量可以被简单理解为多维数组。但张量在TensorFlow中的实现并不是直接采用数组的形式，它只是对TensorFlow中运算结果的引用。在张量中并没有真正保存数字，它保存的是如何得到这些数字的计算过程。 


	import tensorflow as tf
	a = tf.constant([1.0, 2.0], name="a")
	b = tf.constant([2.0, 3.0], name="b")
	result = a + b
	print（result）    # 输出“Tensor("add:0", shape=(2,), dtype=float32) ”

一个张量中主要保存了三个属性：名字（name）、维度（shape）和类型（type）。张量的第一个属性名字不仅是一个张量的唯一标识符，它同样也给出了这个张量是如何计算出来的。张量的命名是通过“node:src_output”的形式来给出。其中node为计算节点的名称，src_output表示当前张量来自节点的第几个输出。比如张量“add:0”就说明了result这个张量是计算节点“add”输出的第一个结果（编号从0开始）。张量的第二个属性是张量的维度（shape）。这个属性描述了一个张量的维度信息。比如“shape=(2,) ”说明了张量result是一个一维数组，这个数组的长度为2。张量的第三个属性是类型（type），每一个张量会有一个唯一的类型。TensorFlow会对参与运算的所有张量进行类型的检查，当发现类型不匹配时会报错。

如果说TensorFlow的第一个词Tensor表明了它的数据结构，那么Flow则体现了它的计算模型。Flow翻译成中文就是“流”，它直观地表达了张量之间通过计算相互转化的过程。

TensorFlow是一个通过计算图的形式来表述计算的编程系统。TensorFlow中的每一个计算都是计算图上的一个节点，而节点之间的边描述了计算之间的依赖关系。TensorFlow计算图定义完成后，我们需要通过会话（Session）来执行定义好的运算。会话拥有并管理TensorFlow程序运行时的所有资源。当所有计算完成之后需要关闭会话来帮助系统回收资源，否则就可能出现资源泄漏的问题。

在TensorFlow中可以通过矩阵乘法的方法实现神经网络的前向传播过程。TensorFlow可以通过变量（tf.Variable）来保存和更新神经网络中的参数。通过满足正太分布的随机数来初始化神经网络中的参数是一个非常常用的方法。为了避免在程序中生成大量常量来提供输入数据，TensorFlow提供了placeholder机制用于提供输入数据。


## MNIST机器学习入门

MNIST好比编程入门的Hello World。 MNIST是一个入门级的计算机视觉数据集，它包含各种手写数字图片及其标签：   

   ![](/images/images_2018/MNIST.png)

上面四张图片的标签分别是5，0，4，1。MNIST包含了60000张图片作为训练数据，10000张图片作为测试数据。

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
	    batch = mnist.train.next_batch(50)
	    sess.run(train_step, feed_dict={x:batch[0], y_:batch[1]})
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

`x = tf.placeholder(tf.float32,[None, 784])`中，x是一个占位符placeholder，表示输入任意数量的MNIST图像，每一张图展平成784维的向量，这里用2维的浮点数张量来表示这些图，张量的形状是[None, 784]，None表示张量的第一个维度可以是任何长度的。虽然placeholder的shape参数是可选的，但有了它，tf能够自动捕捉因数据维度不一致导致的错误。

模型的权重值和偏置量，用Variable表示。一个Variable代表tf计算图中的一个值，能够在计算过程中使用，甚至修改。这里用全为零的张量来初始化W和b：   
`W = tf.Variable(tf.zeros([784,10]))`    
`b = tf.Variable(tf.zeros([10]))`。

占位符是定义一个**可变的常量**，占位符赋值后不用初始化就可以获取值。变量必须初始化才有具体的值，在机器学习中，模型参数一般用变量来表示。


#### 模型的实现和训练

模型的实现只用了一行代码：`y = tf.nn.softmax(tf.matmul(x,W) + b)`。为了训练模型，需要定义一个指标来评估模型的好坏，在机器学习中这个指标称为损失。我们需要尽量最小化损失。交叉熵(cross-entropy)是一个非常常见的损失函数：


   ![](/images/images_2018/cross-entropy.png)

为了计算交叉熵，需要添加一个新的占位符y_，用来输入正确值。计算交叉熵的代码为`cross_entropy = -tf.reduce_sum(y_ * tf.log(y))`。注意交叉熵衡量的不是单一样本的损失，而是整个minibatch的损失。接下来，我们用梯度下降算法以0.01的学习速率最小化交叉熵。在运行计算前，需要初始化我们创建的变量：`init = tf.initialize_all_variables()`。

现在可以在一个Session里启动模型`sess = tf.Session()`，并初始化变量`sess.run(init)`。然后开始训练模型，使用的是小批量梯度下降算法，一次处理100个数据点，循环1000次。

#### 模型的评估

tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，比如`tf.argmax(y,1)`返回的是模型对于任一输入x预测到的标签值，而 `tf.argmax(y_,1)` 代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)：   
`tf.equal(tf.argmax(y,1), tf.argmax(y_,1))`。

代码会返回一组布尔值，我们把布尔值转换成浮点数，然后取平均值，例如[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75：   
`tf.reduce_mean(tf.cast(correct_prediction,"float"))`。

最后计算模型在测试数据集上的正确率：   
`sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels})`

执行结果为：0.9198。

## 深入MNIST

Softmax回归模型在MNIST上只有91%的正确率，这个结果不太好。下面我们将用卷积神经网络来改善效果。这个模型会达到大概99%的正确率，虽然不是最高的，但还是比较让人满意。完整代码如下：

	import tensorflow as tf
	import tensorflow.examples.tutorials.mnist.input_data as input_data
	
	
	def weight_variable(shape):
	  initial = tf.truncated_normal(shape, stddev=0.1)
	  return tf.Variable(initial)
	
	def bias_variable(shape):
	  initial = tf.constant(0.1, shape=shape)
	  return tf.Variable(initial)
	
	def conv2d(x, W):
	  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
	
	def max_pool_2x2(x):
	  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
	                        strides=[1, 2, 2, 1], padding='SAME')
	
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	sess  = tf.InteractiveSession()

	x = tf.placeholder(tf.float32,[None, 784])
	y_ = tf.placeholder("float",[None,10])
	
    #第一层卷积
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	
	x_image = tf.reshape(x, [-1,28,28,1])
	
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
    
    #第二层卷积
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

    #全连接层
	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])
	
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder("float")
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])

	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

	cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.global_variables_initializer())

	for i in range(20000):
	  batch = mnist.train.next_batch(50)
	  if i%100 == 0:
	    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
	    print("step %d, training accuracy %g"%(i, train_accuracy))
	  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	print("test accuracy %g"%accuracy.eval(feed_dict={
	    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

### 权重初始化、卷积和池化

用tf.truncated_normal生成正态分布的数据，作为W的初始值，参数shape表示生成张量的维度，参数stddev表示正态分布的标准差。   
用tf.constant创建了一个常量tensor。创建卷积神经网络，由于使用的是ReLU激励函数，比较好的做好是用一个较小的正数来初始化偏置量，这里使用的是0.1。    
卷积使用1步长，same padding的filter，保证输出和输入是同一大小。  
池化用简单的2*2的filter做max pooling。

### 第一层卷积

第一层卷积由一个卷积接一个max pooling完成。为了用这一层，把x转换成一个4d变量，第2、3维对应图片的宽、高，最后一维代表图片的颜色通道（灰色为1，rgb彩色图则为3）。卷积的权重张量形状是[5,5,1,32]，32是通道数量。

第二层卷积类似，有64个通道。



参考链接：   
http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html










