---
layout:     post
title:      深度神经网络超参数调优编程作业(二)  
keywords:   博客
categories: [机器学习]
tags:	    [深度神经网络，超参数调优]
---

深度学习模型具有非常大的灵活性，但是如果训练数据不够多，可能会产生一个严重的问题：过度拟合，即在训练集上表现很好，但泛化能力比较差。 


# 正则化

你将学习如何在深度学习模型中使用正则化。

首先导入需要用到的库文件。 


	# import packages
	import numpy as np
	import matplotlib.pyplot as plt
	from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
	from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
	import sklearn
	import sklearn.datasets
	import scipy.io
	from testCases import *
	
	%matplotlib inline
	plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
	plt.rcParams['image.interpolation'] = 'nearest'
	plt.rcParams['image.cmap'] = 'gray'


**问题陈述**： 你刚刚被法国足球公司聘为AI专家。他们希望你推荐法国队的守门员应该踢球的位置，以便法国队的球员可以用他们的头来击球。

   ![](/images/images_2018/field_kiank.png) 

<center> <u> **图 1** </u>: **足球场**<br> 守门员将球踢到空中，每只球队的球员都在用头撞击球 </center>


他们为你提供了法国过去10场比赛的2D数据集。
 

	train_X, train_Y, test_X, test_Y = load_2D_dataset()

   ![](/images/images_2018/7-25_01.png)   

每个点对应于足球场上的位置，在法国守门员从足球场的左侧射球之后，足球运动员用他的头击球。  

- 如果点是蓝点，则意味着法国球员击球。
- 如果点是红点，则意味着其它球队的球员击球。

**你的目标**： 使用深度学习模型找到守门员应该踢球的位置。  

**数据集分析**： 这个数据集有点嘈杂，看起来一条对角线较好的将它分成了左上半部分(蓝色)和右下半部分(红色)。

你将首先尝试非正规化模型，然后你将学习如何规范它，并决定选择哪种模式来解决法国足球公司的问题。   




## 1 - 非正则化模型

你将使用以下神经网络（已经实现了），它可以在以下模式使用：

- 正则化模式(*regularization mode*) -- 输入参数 `lambd` 设置为非0的值。在Python中，我们用"`lambd`" 代替 "`lambda`" 因为 "`lambda`" 是Python的保留关键字。 
- Dropout模式(*dropout mode*) -- 输入参数 `keep_prob` 设置为小于1的一个值。   

你将首先尝试没有任何正则化的模型，然后，你将实现：  

- *L2正则化* 函数： `compute_cost_with_regularization()`和  `backward_propagation_with_regularization()` 
- *Dropout* 函数：`forward_propagation_with_dropout()` 和 `back_propagation_with_dropout()`

-----------------------------------

	def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
	    """
	    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
	    
	    Arguments:
	    X -- input data, of shape (input size, number of examples)
	    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
	    learning_rate -- learning rate of the optimization
	    num_iterations -- number of iterations of the optimization loop
	    print_cost -- If True, print the cost every 10000 iterations
	    lambd -- regularization hyperparameter, scalar
	    keep_prob - probability of keeping a neuron active during drop-out, scalar.
	    
	    Returns:
	    parameters -- parameters learned by the model. They can then be used to predict.
	    """
	        
	    grads = {}
	    costs = []                            # to keep track of the cost
	    m = X.shape[1]                        # number of examples
	    layers_dims = [X.shape[0], 20, 3, 1]
	    
	    # Initialize parameters dictionary.
	    parameters = initialize_parameters(layers_dims)
	
	    # Loop (gradient descent)
	
	    for i in range(0, num_iterations):
	
	        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
	        if keep_prob == 1:
	            a3, cache = forward_propagation(X, parameters)
	        elif keep_prob < 1:
	            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
	        
	        # Cost function
	        if lambd == 0:
	            cost = compute_cost(a3, Y)
	        else:
	            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
	            
	        # Backward propagation.
	        assert(lambd==0 or keep_prob==1)    # it is possible to use both L2 regularization and dropout, 
	                                            # but this assignment will only explore one at a time
	        if lambd == 0 and keep_prob == 1:
	            grads = backward_propagation(X, Y, cache)
	        elif lambd != 0:
	            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
	        elif keep_prob < 1:
	            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
	        
	        # Update parameters.
	        parameters = update_parameters(parameters, grads, learning_rate)
	        
	        # Print the loss every 10000 iterations
	        if print_cost and i % 10000 == 0:
	            print("Cost after iteration {}: {}".format(i, cost))
	        if print_cost and i % 1000 == 0:
	            costs.append(cost)
	    
	    # plot the cost
	    plt.plot(costs)
	    plt.ylabel('cost')
	    plt.xlabel('iterations (x1,000)')
	    plt.title("Learning rate =" + str(learning_rate))
	    plt.show()
	    
	    return parameters

我们首先用非正则化的方法训练模型，查看训练集/测试集的准确度。  

	parameters = model(train_X, train_Y)
	print ("On the training set:")
	predictions_train = predict(train_X, train_Y, parameters)
	print ("On the test set:")
	predictions_test = predict(test_X, test_Y, parameters)

   ![](/images/images_2018/7-25_02.png)    

运行结果： 

On the training set:   
Accuracy: 0.947867298578   
On the test set:   
Accuracy: 0.915   

运行下面的代码画出模型的决策边界：  

   ![](/images/images_2018/7-25_03.png) 

可以看到非正则化模型显然过度拟合了训练集，适应了嘈杂的点。接下来让我们看看两种减少过度拟合的方法。


## 2 - L2 正则化

避免过拟合的一种标准方法是 **L2 regularization**。代价函数有了变化，原公式：   
$$J = -\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small  y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} \tag{1}$$   
修改后的公式：   
$$J_{regularized} = \small \underbrace{-\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} }_\text{cross-entropy cost} + \underbrace{\frac{1}{m} \frac{\lambda}{2} \sum\limits_l\sum\limits_k\sum\limits_j W_{k,j}^{[l]2} }_\text{L2 regularization cost} \tag{2}$$


**练习**: 完成 `compute_cost_with_regularization()` 函数。计算 $\sum\limits_k\sum\limits_j W_{k,j}^{[l]2}$  ，可使用：  
```
np.sum(np.square(Wl))   
```    
你需要对 $W^{[1]}$, $W^{[2]}$ 和 $W^{[3]}$ 执行上述操作，并将它们相加，然后乘以 $ \frac{1}{m} \frac{\lambda}{2} $。  

	# GRADED FUNCTION: compute_cost_with_regularization
	
	def compute_cost_with_regularization(A3, Y, parameters, lambd):
	    """
	    Implement the cost function with L2 regularization. See formula (2) above.
	    
	    Arguments:
	    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
	    Y -- "true" labels vector, of shape (output size, number of examples)
	    parameters -- python dictionary containing parameters of the model
	    
	    Returns:
	    cost - value of the regularized loss function (formula (2))
	    """
	    m = Y.shape[1]
	    W1 = parameters["W1"]
	    W2 = parameters["W2"]
	    W3 = parameters["W3"]
	    
	    cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost
	    
	    ### START CODE HERE ### (approx. 1 line)
	    L2_regularization_cost = (1 / m) * (lambd / 2) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
	    ### END CODER HERE ###
	    
	    cost = cross_entropy_cost + L2_regularization_cost
	    
	    return cost


	A3, Y_assess, parameters = compute_cost_with_regularization_test_case()
	
	print("cost = " + str(compute_cost_with_regularization(A3, Y_assess, parameters, lambd = 0.1)))

执行结果：  

cost = 1.78648594516  

代价函数变了，你也必须改变反向传播，根据新的代价函数计算所有梯度。 

**练习**： 实现反向传播的正则化，这些变化涉及dW1, dW2 和 dW3。 对于上述每项，必须添加正则化项($\frac{d}{dW} ( \frac{1}{2}\frac{\lambda}{m}  W^2) = \frac{\lambda}{m} W$)。


	# GRADED FUNCTION: backward_propagation_with_regularization
	
	def backward_propagation_with_regularization(X, Y, cache, lambd):
	    """
	    Implements the backward propagation of our baseline model to which we added an L2 regularization.
	    
	    Arguments:
	    X -- input dataset, of shape (input size, number of examples)
	    Y -- "true" labels vector, of shape (output size, number of examples)
	    cache -- cache output from forward_propagation()
	    lambd -- regularization hyperparameter, scalar
	    
	    Returns:
	    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
	    """
	    
	    m = X.shape[1]
	    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
	    
	    dZ3 = A3 - Y
	    
	    ### START CODE HERE ### (approx. 1 line)
	    dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd / m) * W3
	    ### END CODE HERE ###
	    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
	    
	    dA2 = np.dot(W3.T, dZ3)
	    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
	    ### START CODE HERE ### (approx. 1 line)
	    dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd / m) * W2
	    ### END CODE HERE ###
	    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
	    
	    dA1 = np.dot(W2.T, dZ2)
	    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
	    ### START CODE HERE ### (approx. 1 line)
	    dW1 = 1./m * np.dot(dZ1, X.T) + (lambd / m) * W1
	    ### END CODE HERE ###
	    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
	    
	    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
	                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
	                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
	    
	    return gradients


	X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()
	
	grads = backward_propagation_with_regularization(X_assess, Y_assess, cache, lambd = 0.7)
	print ("dW1 = "+ str(grads["dW1"]))
	print ("dW2 = "+ str(grads["dW2"]))
	print ("dW3 = "+ str(grads["dW3"]))

执行结果/预期结果：  

dW1 = [[-0.25604646  0.12298827 -0.28297129]    
 [-0.17706303  0.34536094 -0.4410571 ]]        
dW2 = [[ 0.79276486  0.85133918]     
 [-0.0957219  -0.01720463]     
 [-0.13100772 -0.03750433]]    
dW3 = [[-1.77691347 -0.11832879 -0.09397446]]  

我们执行model()函数，使用 L2 正则化方法 $(\lambda = 0.7)$。在  `model()` 函数中调用： 
- `compute_cost_with_regularization` ，替代 `compute_cost`
- `backward_propagation_with_regularization` 替代 `backward_propagation`。 


	parameters = model(train_X, train_Y, lambd = 0.7)
	print ("On the train set:")
	predictions_train = predict(train_X, train_Y, parameters)
	print ("On the test set:")
	predictions_test = predict(test_X, test_Y, parameters)

执行结果：  

   ![](/images/images_2018/7-25_04.png)     

On the train set:    
Accuracy: 0.938388625592    
On the test set:    
Accuracy: 0.93    

训练集的准确度增加到93%，解决了过拟合的问题，决策边界见下图: 

   ![](/images/images_2018/7-25_05.png) 

**观察**：

- $\lambda$ 是一个超参数，你可以使用开发集进行调整。
- L2 正则化 使你的决策边界更加平滑。如果 $\lambda$ 太大，也可能“过度平滑”，导致模型具有高偏差。

**L2正则化实际上做什么？**:

L2正则化依赖于这样的假设：具有小权重的模型比具有大权重的模型更简单。因此，通过惩罚代价函数中权重的平方值，可以将所有权重变为更小的值。拥有大权重的代价太大了，这会导致更平滑的模型，在模型中输出随着输入变化的变化会更慢。


**<font color='blue'>划重点</font>** <font color='blue'>--  L2正则化在以下方面有影响：</font>

- <font color='blue'>代价函数的计算公式：</font>
    - <font color='blue'>添加一个正则化项。</font>
- <font color='blue'>反向传播函数：</font>
    - <font color='blue'>权重矩阵中，梯度的计算公式中添加额外的项。</font>
- <font color='blue'>权重衰减("weight decay")： </font>
    - <font color='blue'>权重的值越来越小。</font>


## 3 - Dropout

最后， **dropout** 是一种广泛使用的正则化技术，专门用于深度学习。
**它会在每次迭代中随机关闭一些神经元。**  被关闭的神经元对迭代的前向和反向传播中的训练没有贡献。

当你关闭一些神经元时，你实际上修改了你的模型。Dropout背后的思想是，在每次迭代中，你训练一个仅使用神经元子集的不同模型。随着dropout，你的神经网络会变得对某些神经元不那么敏感，因为它们可能随时被关闭。  


### 3.1 - 使用dropout的前向传播

**练习**：使用dropout实现前向传播。你正在使用3层神经网络，并将dropout添加到第一个和第二个隐藏层。我们不会将dropout应用到输入层和输出层。

**说明**：  
你想关闭第一层和第二层中的一些神经元，为此，你将执行4个步骤：   

1. 你将使用向量化方式，创建一个随机矩阵 $D^{[1]} = [d^{[1](1)} d^{[1](2)} ... d^{[1](m)}] $ ，与$A^{[1]}$相同维度。请使用 `np.random.rand()` 来随机获取0~1中间的数值。

2. 对 $D^{[1]}$ 中的值进行阈值处理，将每个项设置为0或1。提示：为了将矩阵X的所有项设置为0（如果小于0.5），或设置为1(如果大于0.5)，你将执行  `X = (X < 0.5)`。 

3. 将 $A^{[1]}$ 设置为 $A^{[1]} * D^{[1]}$。你可以将 $D^{[1]}$ 视为掩码，这样当它与另一个矩阵相乘时，它会关闭一些值。

4. $A^{[1]}$ 除以 `keep_prob`。这样做，你可以确保代价函数的结果仍然具有不使用dropout时相同的预期值。(这种技术也被称为反向dropout)


	# GRADED FUNCTION: forward_propagation_with_dropout
	
	def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
	    """
	    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
	    
	    Arguments:
	    X -- input dataset, of shape (2, number of examples)
	    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
	                    W1 -- weight matrix of shape (20, 2)
	                    b1 -- bias vector of shape (20, 1)
	                    W2 -- weight matrix of shape (3, 20)
	                    b2 -- bias vector of shape (3, 1)
	                    W3 -- weight matrix of shape (1, 3)
	                    b3 -- bias vector of shape (1, 1)
	    keep_prob - probability of keeping a neuron active during drop-out, scalar
	    
	    Returns:
	    A3 -- last activation value, output of the forward propagation, of shape (1,1)
	    cache -- tuple, information stored for computing the backward propagation
	    """
	    
	    np.random.seed(1)
	    
	    # retrieve parameters
	    W1 = parameters["W1"]
	    b1 = parameters["b1"]
	    W2 = parameters["W2"]
	    b2 = parameters["b2"]
	    W3 = parameters["W3"]
	    b3 = parameters["b3"]
	    
	    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
	    Z1 = np.dot(W1, X) + b1
	    A1 = relu(Z1)
	    ### START CODE HERE ### (approx. 4 lines)         # Steps 1-4 below correspond to the Steps 1-4 described above. 
	    D1 = np.random.rand(A1.shape[0],A1.shape[1])                                        # Step 1: initialize matrix D1 = np.random.rand(..., ...)
	    D1 = (D1 < keep_prob)                                        # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
	    A1 = A1 * D1                                         # Step 3: shut down some neurons of A1
	    A1 = A1 /  keep_prob                                        # Step 4: scale the value of neurons that haven't been shut down
	    ### END CODE HERE ###
	    Z2 = np.dot(W2, A1) + b2
	    A2 = relu(Z2)
	    ### START CODE HERE ### (approx. 4 lines)
	    D2 = np.random.rand(A2.shape[0],A2.shape[1])                                          # Step 1: initialize matrix D2 = np.random.rand(..., ...)
	    D2 = (D2 < keep_prob)                                           # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
	    A2 = A2 * D2                                         # Step 3: shut down some neurons of A2
	    A2 = A2 /  keep_prob                                          # Step 4: scale the value of neurons that haven't been shut down
	    ### END CODE HERE ###
	    Z3 = np.dot(W3, A2) + b3
	    A3 = sigmoid(Z3)
	    
	    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
	    
	    return A3, cache


	X_assess, parameters = forward_propagation_with_dropout_test_case()
	
	A3, cache = forward_propagation_with_dropout(X_assess, parameters, keep_prob = 0.7)
	print ("A3 = " + str(A3))

执行结果： 

A3 = [[ 0.36974721  0.00305176  0.04565099  0.49683389  0.36974721]]

### 3.2 - 使用dropout的反向传播

**Exercise**: Implement the backward propagation with dropout. As before, you are training a 3 layer network. Add dropout to the first and second hidden layers, using the masks $D^{[1]}$ and $D^{[2]}$ stored in the cache. 

**Instruction**:
Backpropagation with dropout is actually quite easy. You will have to carry out 2 Steps:
1. You had previously shut down some neurons during forward propagation, by applying a mask $D^{[1]}$ to `A1`. In backpropagation, you will have to shut down the same neurons, by reapplying the same mask $D^{[1]}$ to `dA1`. 
2. During forward propagation, you had divided `A1` by `keep_prob`. In backpropagation, you'll therefore have to divide `dA1` by `keep_prob` again (the calculus interpretation is that if $A^{[1]}$ is scaled by `keep_prob`, then its derivative $dA^{[1]}$ is also scaled by the same `keep_prob`).

	# GRADED FUNCTION: backward_propagation_with_dropout
	
	def backward_propagation_with_dropout(X, Y, cache, keep_prob):
	    """
	    Implements the backward propagation of our baseline model to which we added dropout.
	    
	    Arguments:
	    X -- input dataset, of shape (2, number of examples)
	    Y -- "true" labels vector, of shape (output size, number of examples)
	    cache -- cache output from forward_propagation_with_dropout()
	    keep_prob - probability of keeping a neuron active during drop-out, scalar
	    
	    Returns:
	    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
	    """
	    
	    m = X.shape[1]
	    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
	    
	    dZ3 = A3 - Y
	    dW3 = 1./m * np.dot(dZ3, A2.T)
	    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
	    dA2 = np.dot(W3.T, dZ3)
	    ### START CODE HERE ### (≈ 2 lines of code)
	    dA2 = dA2 * D2             # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
	    dA2 = dA2 / keep_prob             # Step 2: Scale the value of neurons that haven't been shut down
	    ### END CODE HERE ###
	    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
	    dW2 = 1./m * np.dot(dZ2, A1.T)
	    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
	    
	    dA1 = np.dot(W2.T, dZ2)
	    ### START CODE HERE ### (≈ 2 lines of code)
	    dA1 = dA1 * D1             # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
	    dA1 = dA1 / keep_prob              # Step 2: Scale the value of neurons that haven't been shut down
	    ### END CODE HERE ###
	    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
	    dW1 = 1./m * np.dot(dZ1, X.T)
	    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
	    
	    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
	                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
	                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
	    
	    return gradients


	X_assess, Y_assess, cache = backward_propagation_with_dropout_test_case()
	
	gradients = backward_propagation_with_dropout(X_assess, Y_assess, cache, keep_prob = 0.8)
	
	print ("dA1 = " + str(gradients["dA1"]))
	print ("dA2 = " + str(gradients["dA2"]))

执行结果： 


dA1 = [[ 0.36544439  0.         -0.00188233  0.         -0.17408748]    
 [ 0.65515713  0.         -0.00337459  0.         -0.        ]]     
dA2 = [[ 0.58180856  0.         -0.00299679  0.         -0.27715731]     
 [ 0.          0.53159854 -0.          0.53159854 -0.34089673]      
 [ 0.          0.         -0.00292733  0.         -0.        ]]    


现在我们执行model()函数 (其中`keep_prob = 0.86`)。这意味着第一层和第二层的神经元有14%的概率被关闭。

- `forward_propagation_with_dropout` 替代 `forward_propagation`。
- `backward_propagation_with_dropout` 替代 `backward_propagation`。 

-----------------------------------------------
	parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)
	
	print ("On the train set:")
	predictions_train = predict(train_X, train_Y, parameters)
	print ("On the test set:")
	predictions_test = predict(test_X, test_Y, parameters)

执行结果：  

   ![](/images/images_2018/7-25_06.png)  

On the train set:    
Accuracy: 0.928909952607    
On the test set:    
Accuracy: 0.95    

Dropout的效果很好，测试集的准确度再次提高（达到95%）！你的模型不会过度拟合训练集，并且在测试集上也表现很好，法国足球队将永远感激你！ 运行以下代码得到决策边界：   

	plt.title("Model with dropout")
	axes = plt.gca()
	axes.set_xlim([-0.75,0.40])
	axes.set_ylim([-0.75,0.65])
	plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)  


   ![](/images/images_2018/7-25_07.png)  

**Note**:
- A **common mistake** when using dropout is to use it both in training and testing. You should use dropout (randomly eliminate nodes) only in training. 
- Deep learning frameworks like [tensorflow](https://www.tensorflow.org/api_docs/python/tf/nn/dropout), [PaddlePaddle](http://doc.paddlepaddle.org/release_doc/0.9.0/doc/ui/api/trainer_config_helpers/attrs.html), [keras](https://keras.io/layers/core/#dropout) or [caffe](http://caffe.berkeleyvision.org/tutorial/layers/dropout.html) come with a dropout layer implementation. Don't stress - you will soon learn some of these frameworks.


**<font color='blue'>划重点：</font>**

- Dropout 是一种正则化技术。
- 你只能在训练期间使用dropout。在测试期间不要使用dropout。
- 在前向传播和反向传播过程中都使用dropout。
- 在训练期间，通过keep\_prob划分每个dropout层，确保与预期相符。例如，如果keep\_prob为0.5，那么我们将平均关闭一半节点，只有剩下的一半对解决方案有贡献。除以0.5相当于乘以2。因此输出具有相同的期望值。

## 4 - 结论

**对比三种模型的结果如下**： 
 

<table> 
    <tr>
        <td>
        **model**
        </td>
        <td>
        **train accuracy**
        </td>
        <td>
        **test accuracy**
        </td>

    </tr>
        <td>
        3-layer NN without regularization
        </td>
        <td>
        95%
        </td>
        <td>
        91.5%
        </td>
    <tr>
        <td>
        3-layer NN with L2-regularization
        </td>
        <td>
        94%
        </td>
        <td>
        93%
        </td>
    </tr>
    <tr>
        <td>
        3-layer NN with dropout
        </td>
        <td>
        93%
        </td>
        <td>
        95%
        </td>
    </tr>
</table>  


请注意，正则化会损害训练集的性能，因为它限制了网络过度适应训练集的能力。但由于它最终会提供更好的测试集的准确度，因此它是对系统有益的。


**<font color='blue'>划重点：</font>**

- <font color='blue'>正则化会帮助你减少过度拟合。</font>
- <font color='blue'>正则化会使你的权重降低到更低的值。</font>
- <font color='blue'>L2 正则化 和 Dropout 是两种非常有效的正则化技术。</font>