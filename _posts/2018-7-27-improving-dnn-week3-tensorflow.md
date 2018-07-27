---
layout:     post
title:      深度神经网络编程作业-Tensorflow  
keywords:   博客
categories: [机器学习]
tags:	    [深度神经网络，Tensorflow]
---

欢迎来到Tensorflow教程！ 在这个笔记本中，您将学习Tensorflow的所有基础知识。 您将实现有用的功能，并使用Numpy绘制与您所做的并行。 您将了解张量和操作是什么，以及如何在计算图中执行它们。

完成此任务后，您还可以使用Tensorflow实现自己的深度学习模型。 实际上，使用我们全新的SIGNS数据集，您将构建一个深度神经网络模型，以便以令人印象深刻的准确度识别从0到5的数字。 


   ![](/images/images_2018/shoushi.png) 


# TensorFlow 教程

欢迎来到本周的编程作业。到目前为止，你总是使用numpy来构建神经网络。现在，我们将引导你完成一个深度学习框架，使你可以更轻松的构建神经网络。TensorFlow, PaddlePaddle, Torch, Caffe, Keras 等机器学习框架可以显著加速你的机器学习开发。所有这些框架都有很多文档，你可以随意阅读。在本次作业中，你将学习如何在 TensorFlow 中执行以下操作： 

- 初始化变量
- 开始自己的会话
- 训练算法 
- 实现一个神经网络

编程框架不仅可以缩短编码时间，而且有时还可以加速代码的优化。

## 1 - 探索 Tensorflow 库

首先，你将导入库：

	import math
	import numpy as np
	import h5py
	import matplotlib.pyplot as plt
	import tensorflow as tf
	from tensorflow.python.framework import ops
	from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
	
	%matplotlib inline
	np.random.seed(1)

现在你已经导入了库，我们将引导你完成其不同的应用程序。你将从一个示例开始，我们为你计算一个训练样本的损失：   

$$loss = \mathcal{L}(\hat{y}, y) = (\hat y^{(i)} - y^{(i)})^2 \tag{1}$$


	y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
	y = tf.constant(39, name='y')                    # Define y. Set to 39
	
	loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss
	
	init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
	                                                 # the loss variable will be initialized and ready to be computed
	with tf.Session() as session:                    # Create a session and print the output
	    session.run(init)                            # Initializes the variables
	    print(session.run(loss))                     # Prints the loss

执行结果：

9


在TensorFlow中编写和运行程序有以下步骤：

1. 创建未使用的张量(变量)。
2. 在这些张量之间写操作。
3. 初始化你的张量。 
4. 创建会话。 
5. 运行会话，它将运行你之前写的操作。

因此，当我们创建损失变量时，我们只是简单的定义了一个损失函数，并没有计算它的值。为了计算损失，我们必须运行 `init=tf.global_variables_initializer()`。它初始化了损失变量。最后我们就能够计算并打印  `loss` 的值。

现在让我们看一个简单的例子。运行下面的单元格： 


	a = tf.constant(2)
	b = tf.constant(10)
	c = tf.multiply(a,b)
	print(c) 

执行结果：  

Tensor("Mul:0", shape=(), dtype=int32)

正如所料，你不会看到20！你得到一个张量，结果是一个没有shape属性的张量，并且类型为int32。你所做的只是放在“计算图”中，但你还没有运行这个计算。为了实际乘以这两个数字，你必须创建一个会话并运行它。

	sess = tf.Session()
	print(sess.run(c))

执行结果： 

20

总结一下， **记住初始化你的变量，创建一个会话并在其中运行操作**。

接下来，你必须了解占位符。占位符是一个对象，其值只能在以后指定。要指定占位符的值，可以使用 "feed dictionary" (`feed_dict` 变量)。 下面，我们为x创建了一个占位符，这允许我们稍后在运行会话时传入一个数字。


	# Change the value of x in the feed_dict
	
	x = tf.placeholder(tf.int64, name = 'x')
	print(sess.run(2 * x, feed_dict = {x: 3}))
	sess.close()

执行结果：

6

当你第一次定义`x`时，你不必为它指定一个值。 占位符只是一个变量，您只能在以后运行会话时分配数据。 我们称之为在运行会话时向这些占位符**提供数据**。

我们这样描述正在发生的事情：当您指定计算所需的操作时，您告诉TensorFlow如何构建计算图。 计算图可以有一些占位符，其值将在稍后指定。 最后，当您运行会话时，您告诉TensorFlow执行计算图。

### 1.1 - 线性函数

让我们通过计算以下等式开始这个编程练习：$ Y = WX + b $，其中$ W $和$ X $是随机矩阵，b是随机向量。

**练习**：计算$ WX + b $，其中$ W，X $和$ b $来自随机正态分布。 W的形状为（4,3），X为（3,1），b为（4,1）。 作为示例，以下是如何定义具有形状（3,1）的常量X：

```
X = tf.constant（np.random.randn（3,1），name =“X”）

```
您可能会发现以下功能有用：

- tf.matmul（...，...）进行矩阵乘法
- tf.add（...，...）进行加法
- np.random.randn（...）随机初始化

-------------------------------------------

	# GRADED FUNCTION: linear_function
	
	def linear_function():
	    """
	    Implements a linear function: 
	            Initializes W to be a random tensor of shape (4,3)
	            Initializes X to be a random tensor of shape (3,1)
	            Initializes b to be a random tensor of shape (4,1)
	    Returns: 
	    result -- runs the session for Y = WX + b 
	    """
	    
	    np.random.seed(1)
	    
	    ### START CODE HERE ### (4 lines of code)
	    X = tf.constant(np.random.randn(3,1), name = "X")
	    W = tf.constant(np.random.randn(4,3), name = "W")
	    b = tf.constant(np.random.randn(4,1), name = "b")
	    Y = tf.constant(np.random.randn(4,1), name = "Y")
	    ### END CODE HERE ### 
	    
	    # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate
	    
	    ### START CODE HERE ###
	    sess = tf.Session()
	    result = sess.run(tf.add(tf.matmul(W,X),b))
	    ### END CODE HERE ### 
	    
	    # close the session 
	    sess.close()
	
	    return result

	print( "result = " + str(linear_function()))

执行结果/预期结果：

result = [[-2.15657382]    
 [ 2.95891446]    
 [-1.08926781]    
 [-0.84538042]]   


### 1.2 - 计算sigmoid 

你刚刚实现了线性函数。 Tensorflow 提供了各种常用的神经网络函数，如 `tf.sigmoid` 和 `tf.softmax`。 此练习，我们可以计算输入的sigmoid函数。 

你将使用占位符变量 `x`完成此练习。 运行会话时，你应该使用feed字典 传入输入 `z`。你将：

(i) 创建一个占位符 `x`    
(ii) 在 `tf.sigmoid`中定义计算sigmoid所需的操作    
(iii) 运行会话 

** 练习 **: 实现下面的sigmoid函数。你需要使用以下内容： 

- `tf.placeholder(tf.float32, name = "...")`
- `tf.sigmoid(...)`
- `sess.run(..., feed_dict = {x: z})`


请注意，在tensorflow中创建和使用会话有两种典型方法：    

**方法 1:**

	sess = tf.Session()    
	# Run the variables initialization (if needed), run the operations          
	result = sess.run(..., feed_dict = {...})    
	sess.close() # Close the session


**方法 2:**

	with tf.Session() as sess: 
	    # run the variables initialization (if needed), run the operations
	    result = sess.run(..., feed_dict = {...})
	    # This takes care of closing the session for you :)

-----------------------------------------

	# GRADED FUNCTION: sigmoid
	
	def sigmoid(z):
	    """
	    Computes the sigmoid of z
	    
	    Arguments:
	    z -- input value, scalar or vector
	    
	    Returns: 
	    results -- the sigmoid of z
	    """
	    
	    ### START CODE HERE ### ( approx. 4 lines of code)
	    # Create a placeholder for x. Name it 'x'.
	    x = tf.placeholder(tf.float32, name = "x")
	
	    # compute sigmoid(x)
	    sigmoid = tf.sigmoid(x)
	
	    # Create a session, and run it. Please use the method 2 explained above. 
	    # You should use a feed_dict to pass z's value to x. 
	    with tf.Session() as sess: 
	        # Run session and call the output "result"
	        result = sess.run(sigmoid, feed_dict = {x:z})
	    
	    ### END CODE HERE ###
	    
	    return result

	print ("sigmoid(0) = " + str(sigmoid(0)))
	print ("sigmoid(12) = " + str(sigmoid(12)))

执行结果/预期结果：

sigmoid(0) = 0.5   
sigmoid(12) = 0.999994 



**<font color='blue'>总结一下，你知道怎么：</font>**  

1. 创建占位符   
2. 定义操作对应的计算图    
3. 创建会话    
4. 在运行会话的过程中，使用占位符变量


### 1.3 -  计算成本

你还可以使用内置函数来计算神经网络的成本。所以不需要编写代码来计算 $a^{[2](i)}$ and $y^{(i)}$ for i=1...m：   
$$ J = - \frac{1}{m}  \sum_{i = 1}^m  \large ( \small y^{(i)} \log a^{ [2] (i)} + (1-y^{(i)})\log (1-a^{ [2] (i)} )\large )\small\tag{2}$$

你可以在tensorflow中用一行代码完成它！

**练习**: 计算cross entropy cost，你可以使用以下函数： 


- `tf.nn.sigmoid_cross_entropy_with_logits(logits = ...,  labels = ...)`

代码输入 `z`，计算 sigmoid (得到 `a`) ，然后计算 cross entropy cost $J$。所有这一切都可以通过调用 `tf.nn.sigmoid_cross_entropy_with_logits` 来完成。

$$- \frac{1}{m}  \sum_{i = 1}^m  \large ( \small y^{(i)} \log \sigma(z^{[2](i)}) + (1-y^{(i)})\log (1-\sigma(z^{[2](i)})\large )\small\tag{2}$$

	# GRADED FUNCTION: cost
	
	def cost(logits, labels):
	    """
	    Computes the cost using the sigmoid cross entropy
	    
	    Arguments:
	    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
	    labels -- vector of labels y (1 or 0) 
	    
	    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels" 
	    in the TensorFlow documentation. So logits will feed into z, and labels into y. 
	    
	    Returns:
	    cost -- runs the session of the cost (formula (2))
	    """
	    
	    ### START CODE HERE ### 
	    # Create the placeholders for "logits" (z) and "labels" (y) (approx. 2 lines)
	    z = tf.placeholder(tf.float32, name = "z")
	    y = tf.placeholder(tf.float32, name = "y")
	    
	    # Use the loss function (approx. 1 line)
	    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)
	    
	    # Create a session (approx. 1 line). See method 1 above.
	    sess = tf.Session()
	    
	    # Run the session (approx. 1 line).
	    cost = sess.run(cost, feed_dict = {z:logits, y:labels})
	    
	    # Close the session (approx. 1 line). See method 1 above.
	    sess.close()
	    
	    ### END CODE HERE ###
	    
	    return cost

	logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
	cost = cost(logits, np.array([0,0,1,1]))
	print ("cost = " + str(cost))
	
执行结果/预期结果：

cost = [ 1.00538719  1.03664088  0.41385433  0.39956614]


### 1.4 - 使用独热(One Hot)编码

很多时候，在深度学习中，你会得到一个数字范围从0到C-1的y向量，其中C是类的数量。如果C是4，那么你可能需要转换以下y向量：

   ![](/images/images_2018/onehot.png)

这被称为一个 "one hot" 编码，因为在转换后的表示中，每列的一个元素恰好设置为1。 要在numpy中进行此转换，你可能需要编写几行代码。在tensorflow中，你可以使用一行代码：   

- tf.one_hot(labels, depth, axis) 

**练习：** 实现下面的函数，取一个标签向量和类 $C$的总数，并返回一个热编码。使用 `tf.one_hot()` 。 


	# GRADED FUNCTION: one_hot_matrix
	
	def one_hot_matrix(labels, C):
	    """
	    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
	                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
	                     will be 1. 
	                     
	    Arguments:
	    labels -- vector containing the labels 
	    C -- number of classes, the depth of the one hot dimension
	    
	    Returns: 
	    one_hot -- one hot matrix
	    """
	    
	    ### START CODE HERE ###
	    
	    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
	    C = tf.constant(C, name = "C")
	    
	    # Use tf.one_hot, be careful with the axis (approx. 1 line)
	    one_hot_matrix = tf.one_hot(labels, C, axis=0)
	    
	    # Create the session (approx. 1 line)
	    sess = tf.Session()
	    
	    # Run the session (approx. 1 line)
	    one_hot = sess.run(one_hot_matrix)
	    
	    # Close the session (approx. 1 line). See method 1 above.
	    sess.close()
	    
	    ### END CODE HERE ###
	    
	    return one_hot


	labels = np.array([1,2,3,0,2,1])
	one_hot = one_hot_matrix(labels, C = 4)
	print ("one_hot = " + str(one_hot))

执行结果/预期结果：

one_hot =     
[[ 0.  0.  0.  1.  0.  0.]    
 [ 1.  0.  0.  0.  0.  1.]    
 [ 0.  1.  0.  0.  1.  0.]    
 [ 0.  0.  1.  0.  0.  0.]]    


### 1.5 - 用zeros and ones函数初始化

现在，你将学习如何初始化零和一的向量。你要调用的函数是`tf.ones（）`。 要使用零进行初始化，可以使用tf.zeros（）代替。 这些函数采用一种形状并返回一个分别为0和1的维度形状的数组。

**练习：**实现下面的函数以获取形状并返回一个数组（形状的尺寸为1）。


	# GRADED FUNCTION: ones
	
	def ones(shape):
	    """
	    Creates an array of ones of dimension shape
	    
	    Arguments:
	    shape -- shape of the array you want to create
	        
	    Returns: 
	    ones -- array containing only ones
	    """
	    
	    ### START CODE HERE ###
	    
	    # Create "ones" tensor using tf.ones(...). (approx. 1 line)
	    ones = tf.ones(shape)
	    
	    # Create the session (approx. 1 line)
	    sess = tf.Session()
	    
	    # Run the session to compute 'ones' (approx. 1 line)
	    ones = sess.run(ones)
	    
	    # Close the session (approx. 1 line). See method 1 above.
	    sess.close()
	    
	    ### END CODE HERE ###
	    return ones

    print ("ones = " + str(ones([3])))


执行结果/预期结果：

ones = [ 1.  1.  1.]

# 2 - 在tensorflow中构建你的第一个神经网络

在这部分任务中，你将使用tensorflow构建神经网络。请记住，实现一个tensorflow模型有两部分：

- 创建计算图(computation graph)
- 运行计算图

让我们深入研究你想要解决的问题！

### 2.0 - 问题陈述： SIGNS 数据集

一天下午，我和一些朋友决定教我们的电脑破译手语。我们花了几个小时在白墙前拍照，想出了以下数据集。现在，你的工作是构建一种算法，以促进从语言障碍者到不懂手语的人的通信。

- **训练集**: 1080 个图片 (64 * 64 像素) 表示从0到5的数字 (每个数字180个图片)。
- **测试集**: 120 个图片 (64 * 64 像素) 表示从0到5的数字 (每个数字20个图片)。

请注意，这是 SIGNS 数据集的子集。 完整的数据集包含更多符号。

以下是每个数字的示例，以及如何解释我们如何表示标签。在我们将图像降低到64 * 64 像素之前，这些是原始图片。  


   ![](/images/images_2018/hands.png)

<center> **图 1**</u>: SIGNS dataset <br>  </center>


运行以下代码以加载数据集。


	# Loading the dataset
	X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

更改下面的index的值，并运行单元格以显示数据集中的一些示例。

	# Example of a picture
	index = 0
	plt.imshow(X_train_orig[index])
	print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

执行结果：

y = 5

   ![](/images/images_2018/7-27_01.png)

按照惯例，你flatten图片数据集，然后将其除以255进行标准化。最重要的是，你将每个标签转换为独热矢量，如图1所示。运行以下单元格来执行此操作。


	# Flatten the training and test images
	X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
	X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
	# Normalize image vectors
	X_train = X_train_flatten/255.
	X_test = X_test_flatten/255.
	# Convert training and test labels to one hot matrices
	Y_train = convert_to_one_hot(Y_train_orig, 6)
	Y_test = convert_to_one_hot(Y_test_orig, 6)
	
	print ("number of training examples = " + str(X_train.shape[1]))
	print ("number of test examples = " + str(X_test.shape[1]))
	print ("X_train shape: " + str(X_train.shape))
	print ("Y_train shape: " + str(Y_train.shape))
	print ("X_test shape: " + str(X_test.shape))
	print ("Y_test shape: " + str(Y_test.shape))

执行结果：

number of training examples = 1080    
number of test examples = 120     
X_train shape: (12288, 1080)    
Y_train shape: (6, 1080)    
X_test shape: (12288, 120)    
Y_test shape: (6, 120)    


请注意 12288 的计算公式是 64×64×3。 每个图片是64 * 64 像素，3表示RGB颜色。在继续之前请确保你理解这些的含义。 

**你的目标**是构建一种能够高精度识别符号的算法。为此，你将构建一个tensorflow模型，该模型与你之前在numpy中为cat识别构建的模型几乎相同（但现在使用softmax输出）。这是一个很好的机会，可以将你的numpy实现与tensorflow实现进行比较。

**模型** 是 *LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX*。 这里将SIGMOID输出层替换成SOFTMAX。你可以把SOFTMAX 看作是 SIGMOID 向多分类的扩展。


### 2.1 - 创建占位符

你的第一个任务是为“X”和“Y”创建占位符。 这将允许您稍后在运行会话时传递你的训练数据。

**练习：**执行以下函数以在tensorflow中创建占位符。 


	# GRADED FUNCTION: create_placeholders
	
	def create_placeholders(n_x, n_y):
	    """
	    Creates the placeholders for the tensorflow session.
	    
	    Arguments:
	    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
	    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
	    
	    Returns:
	    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
	    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
	    
	    Tips:
	    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
	      In fact, the number of examples during test/train is different.
	    """
	
	    ### START CODE HERE ### (approx. 2 lines)
	    X = tf.placeholder(tf.float32, (n_x, None), name = "X")
	    Y = tf.placeholder(tf.float32, (n_y, None), name = "Y")
	    ### END CODE HERE ###
    
        return X, Y

	X, Y = create_placeholders(12288, 6)
	print ("X = " + str(X))
	print ("Y = " + str(Y))

执行结果：

X = Tensor("X_1:0", shape=(12288, ?), dtype=float32)   
Y = Tensor("Y_1:0", shape=(6, ?), dtype=float32)


### 2.2 - 初始化参数

你的第二个任务是初始化tensorflow中的参数。

**练习：** 执行以下函数初始化tensorflow中的参数。你将使用 Xavier Initialization 进行权重和零偏置初始化。为了帮助你，对于 W1 和 b1 你可以使用： 


	W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
	b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())

请使用 `seed = 1` 以确保你的结果和我们的结果相符。


	# GRADED FUNCTION: initialize_parameters
	
	def initialize_parameters():
	    """
	    Initializes parameters to build a neural network with tensorflow. The shapes are:
	                        W1 : [25, 12288]
	                        b1 : [25, 1]
	                        W2 : [12, 25]
	                        b2 : [12, 1]
	                        W3 : [6, 12]
	                        b3 : [6, 1]
	    
	    Returns:
	    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
	    """
	    
	    tf.set_random_seed(1)                   # so that your "random" numbers match ours
	        
	    ### START CODE HERE ### (approx. 6 lines of code)
	    W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
	    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
	    W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
	    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
	    W3 = tf.get_variable("W3", [6,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
	    b3 = tf.get_variable("b3", [6,1], initializer = tf.zeros_initializer())
	    ### END CODE HERE ###
	
	    parameters = {"W1": W1,
	                  "b1": b1,
	                  "W2": W2,
	                  "b2": b2,
	                  "W3": W3,
	                  "b3": b3}
	    
	    return parameters
	
	tf.reset_default_graph()
	with tf.Session() as sess:
	    parameters = initialize_parameters()
	    print("W1 = " + str(parameters["W1"]))
	    print("b1 = " + str(parameters["b1"]))
	    print("W2 = " + str(parameters["W2"]))
	    print("b2 = " + str(parameters["b2"]))


执行结果/预期结果:

W1 = <tf.Variable 'W1:0' shape=(25, 12288) dtype=float32_ref>   
b1 = <tf.Variable 'b1:0' shape=(25, 1) dtype=float32_ref>    
W2 = <tf.Variable 'W2:0' shape=(12, 25) dtype=float32_ref>    
b2 = <tf.Variable 'b2:0' shape=(12, 1) dtype=float32_ref>    

正如所料，参数尚未赋值。
	
	
### 2.3 -  tensorflow中的前向传播 

你现在将在tensorflow中实现前向传播。该函数将接收参数字典，它将完成前向传播。你将使用以下函数来完成任务： 

- `tf.add(...,...)` 加法
- `tf.matmul(...,...)` 矩阵乘法
- `tf.nn.relu(...)` 实现ReLU激励函数

**问题：** 实现神经网络的前向传播。我们提供了numpy中的实现，以便你可以将tensorflow中的实现与numpy进行比较。值得注意的是，前向传播在 `z3`处停止。原因在于，tensorflow中最后的线性层输出被作为代价函数的输入。因此你不需要 `a3`!

	# GRADED FUNCTION: forward_propagation
	
	def forward_propagation(X, parameters):
	    """
	    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
	    
	    Arguments:
	    X -- input dataset placeholder, of shape (input size, number of examples)
	    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
	                  the shapes are given in initialize_parameters
	
	    Returns:
	    Z3 -- the output of the last LINEAR unit
	    """
	    
	    # Retrieve the parameters from the dictionary "parameters" 
	    W1 = parameters['W1']
	    b1 = parameters['b1']
	    W2 = parameters['W2']
	    b2 = parameters['b2']
	    W3 = parameters['W3']
	    b3 = parameters['b3']
	    
	    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
	    Z1 = tf.add(tf.matmul(W1,X),b1)                                              # Z1 = np.dot(W1, X) + b1
	    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
	    Z2 = tf.add(tf.matmul(W2,A1),b2)                                                # Z2 = np.dot(W2, a1) + b2
	    A2 = tf.nn.relu(Z2)                                                 # A2 = relu(Z2)
	    Z3 = tf.add(tf.matmul(W3,A2),b3)                                                # Z3 = np.dot(W3,Z2) + b3
	    ### END CODE HERE ###
	    
	    return Z3

	tf.reset_default_graph()
	
	with tf.Session() as sess:
	    X, Y = create_placeholders(12288, 6)
	    parameters = initialize_parameters()
	    Z3 = forward_propagation(X, parameters)
	    print("Z3 = " + str(Z3))

执行结果：

Z3 = Tensor("Add_2:0", shape=(6, ?), dtype=float32)

你可能已经注意到前向传播不会输出任何缓存。下面我们进行反向传播，你会明白其中的原因。

### 2.4 计算代价

如前所述，使用以下方法计算cost非常容易：
```
tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = ..., labels = ...))   
```

**问题**：执行以下代价函数。
 
- 重要的是知道`tf.nn.softmax_cross_entropy_with_logits`的输入参数"`logits`" 和 "`labels`" (样本的数量, 分类的数量)的形状。 我们为你转换了 Z3 和 Y 。
- 此外， `tf.reduce_mean` 是对所有的样本求平均值。

---------------------------------

	# GRADED FUNCTION: compute_cost 
	
	def compute_cost(Z3, Y):
	    """
	    Computes the cost
	    
	    Arguments:
	    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
	    Y -- "true" labels vector placeholder, same shape as Z3
	    
	    Returns:
	    cost - Tensor of the cost function
	    """
	    
	    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
	    logits = tf.transpose(Z3)
	    labels = tf.transpose(Y)
	    
	    ### START CODE HERE ### (1 line of code)
	    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits , labels = labels ))
	    ### END CODE HERE ###
	    
	    return cost

	tf.reset_default_graph()
	
	with tf.Session() as sess:
	    X, Y = create_placeholders(12288, 6)
	    parameters = initialize_parameters()
	    Z3 = forward_propagation(X, parameters)
	    cost = compute_cost(Z3, Y)
	    print("cost = " + str(cost))

执行结果/预期结果：

cost = Tensor("Mean:0", shape=(), dtype=float32)


### 2.5 - 反向传播和参数更新

这是你需要感激编程框架的地方。所有的反向传播和参数更新都在一行代码中处理。将该行代码合入到模型中非常容易。

计算代价函数后，你将创建一个"`optimizer`" 对象。运行tf.session时，必须将此对象与代价函数一起调用。调用时，它将使用所选方法和学习速率对给定的代价函数执行优化。

例如，对于梯度下降，优化器是：   
```
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
```

要进行优化，你需要执行：
```
_ , c = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
```

以相反的顺序通过tensorflow来计算反向传播，从cost到inputs。

**注意** 编码时，我们经常使用 `_` 作为 "一次性" 变量来存储我们以后不需要的值。这里  `_` 表示`optimizer`的求值 ( `c` 取值为 `cost` 变量的值)。

### 2.6 - 构建模型

现在，你需要把它们全部结合在一起！ 

**练习：** 实现模型。你将调用之前实现的功能。


运行以下单元格来训练你的模型！在我们的机器上大概需要5分钟。你迭代100次以后的成本大概是1.016458。如果不是，不要浪费时间，中断训练，并尝试更正你的代码。如果是正确的，请休息一下，5分钟后回来！


    parameters = model(X_train, Y_train, X_test, Y_test)


**Insights**:

- 你的模型看起来足够大，足以适应训练集。但是考虑到训练集和测试集上的准确度差异，你可以尝试添加L2或dropout正则化以减少过度拟合。
- 将会话视为训练模型的代码块。每次在小批量上运行会话时，它都会训练参数。总共运行会话很多次（1500个迭代）0，直到你获得训练有素的参数。