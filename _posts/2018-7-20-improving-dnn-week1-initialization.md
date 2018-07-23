---
layout:     post
title:      深度神经网络超参数调优编程作业(一)  
keywords:   博客
categories: [机器学习]
tags:	    [深度神经网络，超参数调优]
---

这是超参数调优的第一个任务。正确规范模型非常重要，因为它能显著改善你的结果。  


   ![](/images/images_2018/7-20_3models.png) 
   
完成作业后，你将：  

 - 了解可以帮助优化你的模型的不同正则化方法
 - 实施随机失活(dropout)，并看到它是如何作用于数据的
 - 认识到没有正则化的模型虽然可能帮助你在训练集上获得更好的准确性，但不一定在测试集上能获得更好的准确性
 - 了解你可以在模型上同时使用随机失活和正则化   


# 初始化

训练你的神经网络需要初始化权重矩阵。如何为一个新的神经网络选择初始化参数呢？通过作业，你会了解不同的初始化参数会导致不同的结果。 

一个好的初始化参数，可以：  

 - 加速梯度下降的收敛
 - 增加梯度下降收敛到一个较低的训练误差和泛化误差的几率

开始，运行以下代码加载相关的库文件和需要分类的数据集。  


	import numpy as np
	import matplotlib.pyplot as plt
	import sklearn
	import sklearn.datasets
	from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
	from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec
	
	%matplotlib inline
	plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
	plt.rcParams['image.interpolation'] = 'nearest'
	plt.rcParams['image.cmap'] = 'gray'
	
	# load image dataset: blue/red dots in circles
	train_X, train_Y, test_X, test_Y = load_dataset()

运行结果：  

   ![](/images/images_2018/7-20_classify.png)   

你需要设计一个分类器将图中的红点和蓝点区分开来。  

## 1 - 神经网络模型

你将在一个3层神经网络模型（已实现）上实验三种不同的权重初始化方法：  

- 零初始化(*Zeros initialization*) --将参数初始化为零 `initialization = "zeros"` 。
- 随机初始化(*Random initialization*) -- 对参数进行随机初始化  `initialization = "random"` 。权重w初始化的值较大。  
- He初始化(*He initialization*) -- 在一定范围内随机初始化权重w的值  `initialization = "he"` 。 主要用在使用Relu作为激活函数的神经网络中。   

**Instructions**: 阅读并执行下面代码。接下来的部分你将在`model()`中完成三种初始化方法。 

	def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
	    """
	    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
	    
	    Arguments:
	    X -- input data, of shape (2, number of examples)
	    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
	    learning_rate -- learning rate for gradient descent 
	    num_iterations -- number of iterations to run gradient descent
	    print_cost -- if True, print the cost every 1000 iterations
	    initialization -- flag to choose which initialization to use ("zeros","random" or "he")
	    
	    Returns:
	    parameters -- parameters learnt by the model
	    """
	        
	    grads = {}
	    costs = [] # to keep track of the loss
	    m = X.shape[1] # number of examples
	    layers_dims = [X.shape[0], 10, 5, 1]
	    
	    # Initialize parameters dictionary.
	    if initialization == "zeros":
	        parameters = initialize_parameters_zeros(layers_dims)
	    elif initialization == "random":
	        parameters = initialize_parameters_random(layers_dims)
	    elif initialization == "he":
	        parameters = initialize_parameters_he(layers_dims)
	
	    # Loop (gradient descent)
	
	    for i in range(0, num_iterations):
	
	        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
	        a3, cache = forward_propagation(X, parameters)
	        
	        # Loss
	        cost = compute_loss(a3, Y)
	
	        # Backward propagation.
	        grads = backward_propagation(X, Y, cache)
	        
	        # Update parameters.
	        parameters = update_parameters(parameters, grads, learning_rate)
	        
	        # Print the loss every 1000 iterations
	        if print_cost and i % 1000 == 0:
	            print("Cost after iteration {}: {}".format(i, cost))
	            costs.append(cost)
	            
	    # plot the loss
	    plt.plot(costs)
	    plt.ylabel('cost')
	    plt.xlabel('iterations (per hundreds)')
	    plt.title("Learning rate =" + str(learning_rate))
	    plt.show()
	    
	    return parameters


## 2 - 零初始化

在一个神经网络中有两类参数需要初始化：

- 权重矩阵 $(W^{[1]}, W^{[2]}, W^{[3]}, ..., W^{[L-1]}, W^{[L]})$
- 偏差向量 $(b^{[1]}, b^{[2]}, b^{[3]}, ..., b^{[L-1]}, b^{[L]})$

**Exercise**: 完成后续的函数，将所有的参数初始化为0。你会发现这样效果不佳，因为会产生对称失效("break symmetry")，不管怎样，让我们实验零初始化并观察会发生什么。使用 np.zeros((..,..))使其具有正确的形状。  

	# GRADED FUNCTION: initialize_parameters_zeros 
	
	def initialize_parameters_zeros(layers_dims):
	    """
	    Arguments:
	    layer_dims -- python array (list) containing the size of each layer.
	    
	    Returns:
	    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
	                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
	                    b1 -- bias vector of shape (layers_dims[1], 1)
	                    ...
	                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
	                    bL -- bias vector of shape (layers_dims[L], 1)
	    """
	    
	    parameters = {}
	    L = len(layers_dims)            # number of layers in the network
	    
	    for l in range(1, L):
	        ### START CODE HERE ### (≈ 2 lines of code)
	        parameters['W' + str(l)] = np.zeros((layers_dims[l],layers_dims[l-1]))
	        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
	        ### END CODE HERE ###
	    return parameters

**Expected Output**:

<table> 
    <tr>
    <td>
    **W1**
    </td>
        <td>
    [[ 0.  0.  0.]
 [ 0.  0.  0.]]
    </td>
    </tr>
    <tr>
    <td>
    **b1**
    </td>
        <td>
    [[ 0.]
 [ 0.]]
    </td>
    </tr>
    <tr>
    <td>
    **W2**
    </td>
        <td>
    [[ 0.  0.]]
    </td>
    </tr>
    <tr>
    <td>
    **b2**
    </td>
        <td>
    [[ 0.]]
    </td>
    </tr>

</table> 

完成零初始化后，运行以下代码，训练你的模型。  


	parameters = model(train_X, train_Y, initialization = "zeros")
	print ("On the train set:")
	predictions_train = predict(train_X, train_Y, parameters)
	print ("On the test set:")
	predictions_test = predict(test_X, test_Y, parameters)


   ![](/images/images_2018/7-20_01.png) 

性能结果非常差，代价函数甚至没有下降，算法的准确率才50%。我们打印预测的详细信息发现，预测值都是0。    

	print ("predictions_train = " + str(predictions_train))
	print ("predictions_test = " + str(predictions_test))

	plt.title("Model with Zeros initialization")
	axes = plt.gca()
	axes.set_xlim([-1.5,1.5])
	axes.set_ylim([-1.5,1.5])
	plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)  

   ![](/images/images_2018/7-20_02.png)  

通常，将所有权重参数初始化为零会导致对称失效，这意味着每层中的每个神经元都学到同样的东西，其网络不比线性分类器（比如逻辑回归）强大。

<font color='blue'>
**划重点**：            
 
  - 权重参数 $W^{[l]}$ 需要进行随机初始化，以避免产生对称失效。   
  - 可以对偏差参数 $b^{[l]}$进行零初始化。只要 $W^{[l]}$是随机初始化的，就能打破对称。  

</font>

-------------------------------


## 3 - 随机初始化

为了打破对称，我们需要随机初始化权重参数。

**Exercise**:   

完成以下函数，随机初始化你的权重参数，并放大10倍(\*10)，将偏差参数设置为0。 使用函数 `np.random.randn(..,..) * 10` 来初始化权重参数，使用`np.zeros((.., ..))` 来初始化偏差参数。 这里我们使用了固定的 `np.random.seed(..)`，无论运行多少次，都能得到相同的初始化参数。    

	# GRADED FUNCTION: initialize_parameters_random
	
	def initialize_parameters_random(layers_dims):
	    """
	    Arguments:
	    layer_dims -- python array (list) containing the size of each layer.
	    
	    Returns:
	    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
	                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
	                    b1 -- bias vector of shape (layers_dims[1], 1)
	                    ...
	                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
	                    bL -- bias vector of shape (layers_dims[L], 1)
	    """
	    
	    np.random.seed(3)               # This seed makes sure your "random" numbers will be the as ours
	    parameters = {}
	    L = len(layers_dims)            # integer representing the number of layers
	    
	    for l in range(1, L):
	        ### START CODE HERE ### (≈ 2 lines of code)
	        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * 10
	        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
	        ### END CODE HERE ###
	
	    return parameters

	parameters = initialize_parameters_random([3, 2, 1])
	print("W1 = " + str(parameters["W1"]))
	print("b1 = " + str(parameters["b1"]))
	print("W2 = " + str(parameters["W2"]))
	print("b2 = " + str(parameters["b2"]))


**Expected Output**:

<table> 
    <tr>
    <td>
    **W1**
    </td>
        <td>
    [[ 17.88628473   4.36509851   0.96497468]
 [-18.63492703  -2.77388203  -3.54758979]]
    </td>
    </tr>
    <tr>
    <td>
    **b1**
    </td>
        <td>
    [[ 0.]
 [ 0.]]
    </td>
    </tr>
    <tr>
    <td>
    **W2**
    </td>
        <td>
    [[-0.82741481 -6.27000677]]
    </td>
    </tr>
    <tr>
    <td>
    **b2**
    </td>
        <td>
    [[ 0.]]
    </td>
    </tr>

</table> 

完成随机初始化后，运行以下代码，训练你的模型。  

	parameters = model(train_X, train_Y, initialization = "random")
	print ("On the train set:")
	predictions_train = predict(train_X, train_Y, parameters)
	print ("On the test set:")
	predictions_test = predict(test_X, test_Y, parameters)

   ![](/images/images_2018/7-20_03.png)  

我们可以看到性能结果比以前好多了，不再全部输出0，看起来已经打破了对称。 

	print (predictions_train)
	print (predictions_test)  
	
	plt.title("Model with large random initialization")
	axes = plt.gca()
	axes.set_xlim([-1.5,1.5])
	axes.set_ylim([-1.5,1.5])
	plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)  


   ![](/images/images_2018/7-20_04.png)    


**Observations**:    

- 代价函数在迭代开始非常高。这是因为权重参数的随机值过大，会导致一些样本最后的激励函数输出结果非常接近0或1，如果这些样本的预测值是0，那么损失非常大。实际上, 当 $\log(a^{[3]}) = \log(0)$, 损失是无穷大的。
- 初始化做得不好，会导致梯度消失/梯度爆炸问题，从而导致降低优化的速度。
- 训练越久效果越好，但是随机初始化值过大会降低优化的速度。

<font color='blue'>

**总结**：               

 - 权重参数的随机初始化值过大，会导致性能结果不佳。 
 - 希望使用较小的随机值会更好。关键问题是：随机值多小合适？
</font>

## 4 - He initialization

最后，实验的是"He Initialization"方法。 this is named for the first author of He et al., 2015. (If you have heard of "Xavier initialization", this is similar except Xavier initialization uses a scaling factor for the weights $W^{[l]}$ of `sqrt(1./layers_dims[l-1])` where He initialization would use `sqrt(2./layers_dims[l-1])`.)

**Exercise**: 使用He初始化方法完成以下函数。

**Hint**: 这个函数类似之前的 `initialize_parameters_random(...)`。唯一的区别是，不再是在  `np.random.randn(..,..)` 的基础上乘以 10，而是乘以 $\sqrt{\frac{2}{\text{dimension of the previous layer}}}$，这种方法通常使用在以Relu作为激活函数的神经网络中。 


	# GRADED FUNCTION: initialize_parameters_he
	
	def initialize_parameters_he(layers_dims):
	    """
	    Arguments:
	    layer_dims -- python array (list) containing the size of each layer.
	    
	    Returns:
	    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
	                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
	                    b1 -- bias vector of shape (layers_dims[1], 1)
	                    ...
	                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
	                    bL -- bias vector of shape (layers_dims[L], 1)
	    """
	    
	    np.random.seed(3)
	    parameters = {}
	    L = len(layers_dims) - 1 # integer representing the number of layers
	     
	    for l in range(1, L + 1):
	        ### START CODE HERE ### (≈ 2 lines of code)
	        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1]) 
	        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
	        ### END CODE HERE ###
	        
	    return parameters

	parameters = initialize_parameters_he([2, 4, 1])
	print("W1 = " + str(parameters["W1"]))
	print("b1 = " + str(parameters["b1"]))
	print("W2 = " + str(parameters["W2"]))
	print("b2 = " + str(parameters["b2"]))

**Expected Output**:

<table> 
    <tr>
    <td>
    **W1**
    </td>
        <td>
    [[ 1.78862847  0.43650985]
 [ 0.09649747 -1.8634927 ]
 [-0.2773882  -0.35475898]
 [-0.08274148 -0.62700068]]
    </td>
    </tr>
    <tr>
    <td>
    **b1**
    </td>
        <td>
    [[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]
    </td>
    </tr>
    <tr>
    <td>
    **W2**
    </td>
        <td>
    [[-0.03098412 -0.33744411 -0.92904268  0.62552248]]
    </td>
    </tr>
    <tr>
    <td>
    **b2**
    </td>
        <td>
    [[ 0.]]
    </td>
    </tr>

</table> 


完成随机初始化后，运行以下代码，训练你的模型。 

	parameters = model(train_X, train_Y, initialization = "he")
	print ("On the train set:")
	predictions_train = predict(train_X, train_Y, parameters)
	print ("On the test set:")
	predictions_test = predict(test_X, test_Y, parameters)

   ![](/images/images_2018/7-20_05.png)  

可以看到准确度高多了，在训练集和测试集上分别达到了99%和96%。  

	plt.title("Model with He initialization")
	axes = plt.gca()
	axes.set_xlim([-1.5,1.5])
	axes.set_ylim([-1.5,1.5])
	plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

   ![](/images/images_2018/7-20_06.png)  

**Observations**:

- 使用He初始化方法的模型，在少量迭代后，能够非常好的区分蓝点和红点。


## 5 - 结论  


你已经看到了三种不同类型的初始化，在相同数量的迭代和相同的超参数的条件下，比较结果如下：  

<table> 
    <tr>
        <td>
        **Model**
        </td>
        <td>
        **Train accuracy**
        </td>
        <td>
        **Problem/Comment**
        </td>

    </tr>
        <td>
        3-layer NN with zeros initialization
        </td>
        <td>
        50%
        </td>
        <td>
        fails to break symmetry
        </td>
    <tr>
        <td>
        3-layer NN with large random initialization
        </td>
        <td>
        83%
        </td>
        <td>
        too large weights 
        </td>
    </tr>
    <tr>
        <td>
        3-layer NN with He initialization
        </td>
        <td>
        99%
        </td>
        <td>
        recommended method
        </td>
    </tr>
</table> 

---------------------------------

<font color='blue'> 

**划重点**：            


 - 不同的初始化方法导致不同的结果
 - 随机初始化可以打破对称，确保不同的隐藏单元可以学习到不同的东西
 - 权重参数的初始化值不宜过大
 - He初始化方法在使用ReLU激励函数的网络中非常有效 