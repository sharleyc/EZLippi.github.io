---
layout:     post
title:      深层神经网络编程作业(三)  
keywords:   博客
categories: [机器学习]
tags:	    [深度学习，深层神经网络]
---

这个作业内容是在前面的基础上构建一个能识别猫的图片的深度神经网络。在之前的练习中，我们使用逻辑回归模型识别猫图片的准确率是68%，而这次的算法准确度能达到80%。这个作业可以帮助你：  

 - 学习如何利用辅助函数去构建一个你想要的模型。
 - 实验不同的模型，并观察它们的行为方式。
 - 在你从头开始构建神经网络之前，构建辅助函数能使你的工作更容易一些。  



## 1 - Packages

完成这个作业需要用到以下Python库，我们首先需要导入它们。
 
- [numpy](www.numpy.org) 
- [matplotlib](http://matplotlib.org) 
- [h5py](http://www.h5py.org) 
- [PIL](http://www.pythonware.com/products/pil/)
- [scipy](https://www.scipy.org/) 
- dnn\_app\_utils
- np.random.seed(1)

------------------------
	import time   
	import numpy as np   
	import h5py
	import matplotlib.pyplot as plt
	import scipy
	from PIL import Image
	from scipy import ndimage
	from dnn_app_utils_v3 import *
	
	%matplotlib inline
	plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
	plt.rcParams['image.interpolation'] = 'nearest'
	plt.rcParams['image.cmap'] = 'gray'
	
	%load_ext autoreload
	%autoreload 2
	
	np.random.seed(1)

---------------------------------

## 2 - Dataset

**Problem Statement**: 会给你一个数据集 ("data.h5") 它包含：  

   - 有标签的训练集图片m_train，1 表示cat，0 表示non-cat
   - 有标签的测试集图片m_test
   - 每张图片的形状是(num_px, num_px, 3)，其中3表示3个颜色通道 (RGB)

我们先了解一下数据集。以下代码的作用是加载数据。

    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()  

以下代码会显示数据集中的一张图片。你可以改变index的值，重新运行代码查看不同的图片。  

	# Example of a picture
	index = 10
	plt.imshow(train_x_orig[index])
	print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.") 

运行结果：  

   ![](/images/images_2018/7-19_01.jpg)    


	# Explore your dataset 
	m_train = train_x_orig.shape[0]
	num_px = train_x_orig.shape[1]
	m_test = test_x_orig.shape[0]
	
	print ("Number of training examples: " + str(m_train))
	print ("Number of testing examples: " + str(m_test))
	print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
	print ("train_x_orig shape: " + str(train_x_orig.shape))
	print ("train_y shape: " + str(train_y.shape))
	print ("test_x_orig shape: " + str(test_x_orig.shape))
	print ("test_y shape: " + str(test_y.shape))

运行结果： 

	Number of training examples: 209
	Number of testing examples: 50
	Each image is of size: (64, 64, 3)
	train_x_orig shape: (209, 64, 64, 3)
	train_y shape: (1, 209)
	test_x_orig shape: (50, 64, 64, 3)
	test_y shape: (1, 50)  

你需要对图片进行处理(reshape, standardize)后，再输入进网络中，代码如下：  

   ![](/images/images_2018/imvectorkiank.png) 


	# Reshape the training and test examples 
	train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
	test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
	
	# Standardize data to have feature values between 0 and 1.
	train_x = train_x_flatten/255.
	test_x = test_x_flatten/255.
	
	print ("train_x's shape: " + str(train_x.shape))
	print ("test_x's shape: " + str(test_x.shape))
 
运行结果：  

	train_x's shape: (12288, 209)
	test_x's shape: (12288, 50) 

说明： 12288 = 64 * 64 * 3，对应变形后的图片向量大小。  

## 3 - Architecture of your model   

现在是时候构建深层神经网络来识别猫的图片了。

首先你需要构建两个不同的模型：

- 一个2层的神经网络
- 一个L层的神经网络

接下来你需要比较不同模型的性能，并且尝试不同的层**L**。   


### 3.1 - 2-layer neural network  


   ![](/images/images_2018/2layerNN_kiank.png)  

<center> <u>图 2</u>: 2-layer neural network. <br> ***INPUT -> LINEAR -> RELU -> LINEAR -> SIGMOID -> OUTPUT***. </center>

<u>关于图2的详细信息</u>:   

- 输入图片是(64,64,3)，将图片扁平化处理后是一个大小是(12288,1)的向量。 
- 权重矩阵$W^{[1]}$ 的size是$(n^{[1]}, 12288)$，权重矩阵和X进行矩阵乘法。
- 然后加上偏差后，应用RELU函数得到对应的a向量: $[a_0^{[1]}, a_1^{[1]},..., a_{n^{[1]}-1}^{[1]}]^T$。
- 重复上述过程。
- 权重矩阵$W^{[2]}$ 和上述结果向量相乘，加上偏差(bias)。
- 最后，在上述结果的基础上, 应用sigmoid函数。如果结果大于 0.5，那么就把它归类为一只猫。

### 3.2 - L-layer deep neural network


这里给出了一个简化的图来表示**L**层的深度神经网络结构：  

  ![](/images/images_2018/LlayerNN_kiank.png)  

<center> <u>图 3</u>: L-layer neural network. <br> ***[LINEAR -> RELU] $\times$ (L-1) -> LINEAR -> SIGMOID***</center>

区别于2层的神经网络结构的是： 

中间循环了L-1次的RELU函数，而不是一次。    

### 3.3 - 一般方法论

你需要遵循以下深度学习的方法论去构建模型： 

    1. 初始化参数 / 定义超参数
    2. 循环num_iterations次：
        a. 前向传播
        b. 计算代价函数
        c. 反向传播
        d. 更新参数 
    3. 使用训练得到的参数来预测标签

现在让我们来完成这两个模型把！  

## 4 - Two-layer neural network

**Question**:  在前面的作业里，你已经使用辅助函数，构建了一个2层的神经网络，遵循以下结构：  *LINEAR -> RELU -> LINEAR -> SIGMOID*  你需要用到的函数以及它们的输入参数如下：   


	def initialize_parameters(n_x, n_h, n_y):     
	    ...
	    return parameters 
	def linear_activation_forward(A_prev, W, b, activation):
	    ...
	    return A, cache
	def compute_cost(AL, Y):
	    ...
	    return cost
	def linear_activation_backward(dA, cache, activation):
	    ...
	    return dA_prev, dW, db
	def update_parameters(parameters, grads, learning_rate):
	    ...
	    return parameters

----------------------- 

	### CONSTANTS DEFINING THE MODEL ####
	n_x = 12288     # num_px * num_px * 3
	n_h = 7
	n_y = 1
	layers_dims = (n_x, n_h, n_y)

	# GRADED FUNCTION: two_layer_model
	
	def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
	    """
	    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
	    
	    Arguments:
	    X -- input data, of shape (n_x, number of examples)
	    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
	    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
	    num_iterations -- number of iterations of the optimization loop
	    learning_rate -- learning rate of the gradient descent update rule
	    print_cost -- If set to True, this will print the cost every 100 iterations 
	    
	    Returns:
	    parameters -- a dictionary containing W1, W2, b1, and b2
	    """
	    
	    np.random.seed(1)
	    grads = {}
	    costs = []                              # to keep track of the cost
	    m = X.shape[1]                           # number of examples
	    (n_x, n_h, n_y) = layers_dims
	    
	    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
	    ### START CODE HERE ### (≈ 1 line of code)
	    parameters = initialize_parameters(n_x, n_h, n_y)
	    ### END CODE HERE ###
	    
	    # Get W1, b1, W2 and b2 from the dictionary parameters.
	    W1 = parameters["W1"]
	    b1 = parameters["b1"]
	    W2 = parameters["W2"]
	    b2 = parameters["b2"]
	    
	    # Loop (gradient descent)
	
	    for i in range(0, num_iterations):
	
	        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
	        ### START CODE HERE ### (≈ 2 lines of code)
	        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
	        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
	        ### END CODE HERE ###
	        
	        # Compute cost
	        ### START CODE HERE ### (≈ 1 line of code)
	        cost =  compute_cost(A2, Y)
	        ### END CODE HERE ###
	        
	        # Initializing backward propagation
	        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
	        
	        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
	        ### START CODE HERE ### (≈ 2 lines of code)
	        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
	        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
	        ### END CODE HERE ###
	        
	        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
	        grads['dW1'] = dW1
	        grads['db1'] = db1
	        grads['dW2'] = dW2
	        grads['db2'] = db2
	        
	        # Update parameters.
	        ### START CODE HERE ### (approx. 1 line of code)
	        parameters = update_parameters(parameters, grads, learning_rate)
	        ### END CODE HERE ###
	
	        # Retrieve W1, b1, W2, b2 from parameters
	        W1 = parameters["W1"]
	        b1 = parameters["b1"]
	        W2 = parameters["W2"]
	        b2 = parameters["b2"]
	        
	        # Print the cost every 100 training example
	        if print_cost and i % 100 == 0:
	            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
	        if print_cost and i % 100 == 0:
	            costs.append(cost)
	       
	    # plot the cost
	
	    plt.plot(np.squeeze(costs))
	    plt.ylabel('cost')
	    plt.xlabel('iterations (per tens)')
	    plt.title("Learning rate =" + str(learning_rate))
	    plt.show()
	    
	    return parameters

    parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)

  ![](/images/images_2018/7-19-2layer-output.png)  

从上图可以看到代价函数一直是下降的。在完成这个作业的时候，在参数初始化这里我犯了一个错误，调用的是L层神经网络的函数，而不是2层神经网络的函数，导致结果不正确。  


**Expected Output**:
<table> 
    <tr>
        <td> **Cost after iteration 0**</td>
        <td> 0.6930497356599888 </td>
    </tr>
    <tr>
        <td> **Cost after iteration 100**</td>
        <td> 0.6464320953428849 </td>
    </tr>
    <tr>
        <td> **...**</td>
        <td> ... </td>
    </tr>
    <tr>
        <td> **Cost after iteration 2400**</td>
        <td> 0.048554785628770206 </td>
    </tr>
</table>


我们使用训练得到的参数来分类训练集里的图片，运行结果：准确率为1.0。    

    predictions_train = predict(train_x, train_y, parameters)    

执行结果： Accuracy: 1.0。  

**Expected Output**:
<table> 
    <tr>
        <td> Accuracy</td>
        <td> 1.0 </td>
    </tr>
</table>

使用训练得到的参数来分类测试集里的图片，运行结果：准确率为0.72。

    predictions_test = predict(test_x, test_y, parameters)  

执行结果： Accuracy: 0.72。 

**Expected Output**:

<table> 
    <tr>
        <td> Accuracy</td>
        <td> 0.72 </td>
    </tr>
</table>  

**Note**: 你也许会发现这个模型执行较少的次数（比如1500次），在测试集上的表现会更好。这种现象被称为"early stopping"，在下一门课程中会讨论它。这是一种防止过拟合的方法。 

2层神经网络的性能表现（72%）看起来比逻辑回归（70%）好，接下来让我们看看L层模型的表现是否更好。  

## 5 - L-layer Neural Network

**Question**: 使用辅助函数，构建一个**L**层的神经网络，遵循以下结构： *[LINEAR -> RELU]$\times$(L-1) -> LINEAR -> SIGMOID*  你需要用到的函数以及它们的输入参数如下：

	def initialize_parameters_deep(layers_dims):
	    ...
	    return parameters 
	def L_model_forward(X, parameters):
	    ...
	    return AL, caches
	def compute_cost(AL, Y):
	    ...
	    return cost
	def L_model_backward(AL, Y, caches):
	    ...
	    return grads
	def update_parameters(parameters, grads, learning_rate):
	    ...
	    return parameters

-------------------------------------------

	### CONSTANTS ###
	layers_dims = [12288, 20, 7, 5, 1] #  4-layer model 



	# GRADED FUNCTION: L_layer_model
	
	def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
	    """
	    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
	    
	    Arguments:
	    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
	    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
	    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
	    learning_rate -- learning rate of the gradient descent update rule
	    num_iterations -- number of iterations of the optimization loop
	    print_cost -- if True, it prints the cost every 100 steps
	    
	    Returns:
	    parameters -- parameters learnt by the model. They can then be used to predict.
	    """
	
	    np.random.seed(1)
	    costs = []                         # keep track of cost
	    
	    # Parameters initialization. (≈ 1 line of code)
	    ### START CODE HERE ###
	    parameters = initialize_parameters_deep(layers_dims)
	    ### END CODE HERE ###
	    
	    # Loop (gradient descent)
	    for i in range(0, num_iterations):
	
	        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
	        ### START CODE HERE ### (≈ 1 line of code)
	        AL, caches = L_model_forward(X, parameters)
	        ### END CODE HERE ###
	        
	        # Compute cost.
	        ### START CODE HERE ### (≈ 1 line of code)
	        cost = compute_cost(AL, Y)
	        ### END CODE HERE ###
	    
	        # Backward propagation.
	        ### START CODE HERE ### (≈ 1 line of code)
	        grads = L_model_backward(AL, Y, caches)
	        ### END CODE HERE ###
	 
	        # Update parameters.
	        ### START CODE HERE ### (≈ 1 line of code)
	        parameters = update_parameters(parameters, grads, learning_rate)
	        ### END CODE HERE ###
	                
	        # Print the cost every 100 training example
	        if print_cost and i % 100 == 0:
	            print ("Cost after iteration %i: %f" %(i, cost))
	        if print_cost and i % 100 == 0:
	            costs.append(cost)
	            
	    # plot the cost
	    plt.plot(np.squeeze(costs))
	    plt.ylabel('cost')
	    plt.xlabel('iterations (per tens)')
	    plt.title("Learning rate =" + str(learning_rate))
	    plt.show()
	    
	    return parameters

现在可以训练你的模型了。

    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

  ![](/images/images_2018/7-19-llayer-output.png)  

**Expected Output**:
<table> 
    <tr>
        <td> **Cost after iteration 0**</td>
        <td> 0.771749 </td>
    </tr>
    <tr>
        <td> **Cost after iteration 100**</td>
        <td> 0.672053 </td>
    </tr>
    <tr>
        <td> **...**</td>
        <td> ... </td>
    </tr>
    <tr>
        <td> **Cost after iteration 2400**</td>
        <td> 0.092878 </td>
    </tr>
</table>  

在训练集的测试结果为0.985645933014，在测试集上的测试结果为0.8。  

    pred_train = predict(train_x, train_y, parameters)  

<table>
    <tr>
    <td>
    **Train Accuracy**
    </td>
    <td>
    0.985645933014
    </td>
    </tr>
</table>

    pred_test = predict(test_x, test_y, parameters)

**Expected Output**:

<table> 
    <tr>
        <td> **Test Accuracy**</td>
        <td> 0.8 </td>
    </tr>
</table>

我们可以看到，在同样的测试集上，4层的神经网络(80%)比2层的神经网络(72%)有更好的性能表现。在接下里的课程中，你们将进一步学习如何通过设置更好的超参数来获得更高的准确度。  

##  6) 结果分析

首先，我们来看看L层模型标记错误的图片。 这里显示了部分标记错误的图片。   

  ![](/images/images_2018/7-19-error_pics.png)   

**模型在以下这些类型的图片上表现比较差:**  


- 猫的身体位置比较特别
- 图片背景色和猫的颜色比较接近
- 猫的颜色或品种比较特别
- 相机角度
- 图片的亮度
- 放大或缩小 (图片中的猫特别大或特别小)    


## 7) 用你自己的图片进行测试

你可以使用自己的图片来测试你的模型。 步骤： 

   - 点击 "File" --> "Open" 
   - 在Jupyter Notebook的"images" 目录中添加你的图片
   - 在下面的代码中修改图片的名字
   - 运行代码，查看算法结果是否正确 (1 = cat, 0 = non-cat)!

---------------------------------------------

	## START CODE HERE ##
	my_image = "cat.jpg" # change this to the name of your image file 
	my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
	## END CODE HERE ##
	
	fname = "images/" + my_image
	image = np.array(ndimage.imread(fname, flatten=False))
	my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
	my_image = my_image/255.
	my_predicted_image = predict(my_image, my_label_y, parameters)
	
	plt.imshow(image)
	print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")


 ![](/images/images_2018/7-19-cat-test.png) 