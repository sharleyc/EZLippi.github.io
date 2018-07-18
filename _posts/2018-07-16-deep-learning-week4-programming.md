---
layout:     post  
title:      Building your Deep Neural Network: Step by Step   
keywords:   博客   
categories: [机器学习]   
tags:	    [深度学习，深层神经网络]  
---

## 1 - 需要的库文件  

Let's first import all the packages that you will need during this assignment.          

-  [numpy](www.numpy.org) 是Python进行科学计算的主要库文件。
-  [matplotlib](http://matplotlib.org) 是Python的2D绘图库，仅需要几行代码，便可以生成出版质量级别的图形。    
- dnn_utils 包含了这次作业需要用到的一些函数定义。  
- testCases 包含了用于验证提交代码正确性的测试用例。
- np.random.seed(1) 用于给作业打分的，请不要改变seed()值。seed()用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed()值，则每次生成的随机数都相同，如果不设置这个值，则系统根据时间自己选择这个值，此时每次生成的随机数因时间差异而不同。 


    import numpy as np   
	import h5py   
	import matplotlib.pyplot as plt    
	from testCases\_v4 import *    
	from dnn\_utils\_v2 import sigmoid, sigmoid_backward, relu, relu\_backward    
	
	%matplotlib inline     
	plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots   
	plt.rcParams['image.interpolation'] = 'nearest'      
	plt.rcParams['image.cmap'] = 'gray'      
	
	%load_ext autoreload
	%autoreload 2
	
	np.random.seed(1)

## 2 - 作业大纲

你需要先完成一些有用函数的编码（会提供详细的说明），这些函数会帮助你完成构建深度神经网络。以下是作业大纲：

- 参数初始化。
- 完成前向传播模块（下图的紫色部分）。
     - 完成前向传播的线性模块(resulting in  $Z^{[l]}$).
     - 我们提供给你激励函数 (relu/sigmoid)。
     - Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
     - Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer **L**). This gives you a new L_model_forward function.
- 计算损失函数。
- 完成后向传播模块（下图的红色部分）.
    - Complet the LINEAR part of a layer's backward propagation step.
    - We give you the gradient of the ACTIVATE function (relu_backward/sigmoid_backward) 
    - Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function.
    - Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
- 更新参数。  

   ![](/images/images_2018/7-13_01.png)         
       
 <center>**Figure 1**</center> 


**Note** 每一个前向传播的函数，都会对应一个后向传播的函数，因此你需要存储一些有用的值，在后向传播中用来计算梯度下降。作业完整显示了如何具体实现这些步骤。

## 3 - 初始化

You will write two helper functions that will initialize the parameters for your model. The first function will be used to initialize parameters for a two layer model. The second one will generalize this initialization process to **L** layers.

### 3.1 - 2-layer Neural Network

**Exercise**: Create and initialize the parameters of the 2-layer neural network.

**Instructions**:
- The model's structure is: *LINEAR -> RELU -> LINEAR -> SIGMOID*. 
- Use random initialization for the weight matrices. Use `np.random.randn(shape)*0.01` with the correct shape.
- Use zero initialization for the biases. Use `np.zeros(shape)`.
     
	# GRADED FUNCTION: initialize_parameters
	
	def initialize_parameters(n_x, n_h, n_y):
	    """
	    Argument:
	    n_x -- size of the input layer
	    n_h -- size of the hidden layer
	    n_y -- size of the output layer
	    
	    Returns:
	    parameters -- python dictionary containing your parameters:
	                    W1 -- weight matrix of shape (n_h, n_x)
	                    b1 -- bias vector of shape (n_h, 1)
	                    W2 -- weight matrix of shape (n_y, n_h)
	                    b2 -- bias vector of shape (n_y, 1)
	    """
	    
	    np.random.seed(1)
	    
	    ### START CODE HERE ### (≈ 4 lines of code)
	    W1 = np.random.randn(n_h,n_x) * 0.01
	    b1 = np.zeros((n_h,1))
	    W2 = np.random.randn(n_y,n_h) * 0.01
	    b2 = np.zeros((n_y,1))
	    ### END CODE HERE ###
	    
	    assert(W1.shape == (n_h, n_x))
	    assert(b1.shape == (n_h, 1))
	    assert(W2.shape == (n_y, n_h))
	    assert(b2.shape == (n_y, 1))
	    
	    parameters = {"W1": W1,
	                  "b1": b1,
	                  "W2": W2,
	                  "b2": b2}
	    
	    return parameters    


		parameters = initialize_parameters(3,2,1)
		print("W1 = " + str(parameters["W1"]))
		print("b1 = " + str(parameters["b1"]))
		print("W2 = " + str(parameters["W2"]))
		print("b2 = " + str(parameters["b2"]))

**Expected output**:
       
<table style="width:80%">
  <tr>
    <td> W1 </td>
    <td> [[ 0.01624345 -0.00611756 -0.00528172]
 [-0.01072969  0.00865408 -0.02301539]] </td> 
  </tr>

  <tr>
    <td> b1 </td>
    <td>[[ 0.]
 [ 0.]]</td> 
  </tr>
  
  <tr>
    <td> W2 </td>
    <td> [[ 0.01744812 -0.00761207]]</td>
  </tr>
  
  <tr>
    <td> b2 </td>
    <td> [[ 0.]] </td> 
  </tr>
</table>

### np.random.randn()  

np.random.randn(d0,d1,...,dn)  

- d0,d1,...,dn表示返回的array的维度，当没有参数时，返回float   
- 返回一个或一组具有标准正态分布的样本，返回值类型为ndarray or float   

        np.random.randn(2,1)   

执行结果：  
        
    array([[0.41230963],
       [1.76974892]]) 

### np.zeros()  

np.zeros(shape, dtype=float, order='C')  

- shape表示返回的array的形状，比如(2,3) or 2 
- dtype表示返回array中的数据的类型，默认是np.float64 
- 返回指定形状的ndarray

        np.zeros((2,1))

执行结果：  

	array([[0.],
	       [0.]])  


### 3.2 - L-layer Neural Network

The initialization for a deeper L-layer neural network is more complicated because there are many more weight matrices and bias vectors. When completing the `initialize_parameters_deep`, you should make sure that your dimensions match between each layer. Recall that $n^{[l]}$ is the number of units in layer $l$. Thus for example if the size of our input $X$ is $(12288, 209)$ (with $m=209$ examples) then:

<table style="width:100%">


    <tr>
        <td>  </td> 
        <td> **Shape of W** </td> 
        <td> **Shape of b**  </td> 
        <td> **Activation** </td>
        <td> **Shape of Activation** </td> 
    <tr>
    
    <tr>
        <td> **Layer 1** </td> 
        <td> $(n^{[1]},12288)$ </td> 
        <td> $(n^{[1]},1)$ </td> 
        <td> $Z^{[1]} = W^{[1]}  X + b^{[1]} $ </td> 
        
        <td> $(n^{[1]},209)$ </td> 
    <tr>
    
    <tr>
        <td> **Layer 2** </td> 
        <td> $(n^{[2]}, n^{[1]})$  </td> 
        <td> $(n^{[2]},1)$ </td> 
        <td>$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$ </td> 
        <td> $(n^{[2]}, 209)$ </td> 
    <tr>
   
       <tr>
        <td> $\vdots$ </td> 
        <td> $\vdots$  </td> 
        <td> $\vdots$  </td> 
        <td> $\vdots$</td> 
        <td> $\vdots$  </td> 
    <tr>
    
   <tr>
        <td> **Layer L-1** </td> 
        <td> $(n^{[L-1]}, n^{[L-2]})$ </td> 
        <td> $(n^{[L-1]}, 1)$  </td> 
        <td>$Z^{[L-1]} =  W^{[L-1]} A^{[L-2]} + b^{[L-1]}$ </td> 
        <td> $(n^{[L-1]}, 209)$ </td> 
    <tr>
    
    
   <tr>
        <td> **Layer L** </td> 
        <td> $(n^{[L]}, n^{[L-1]})$ </td> 
        <td> $(n^{[L]}, 1)$ </td>
        <td> $Z^{[L]} =  W^{[L]} A^{[L-1]} + b^{[L]}$</td>
        <td> $(n^{[L]}, 209)$  </td> 
    <tr>

</table>

Remember that when we compute $W X + b$ in python, it carries out broadcasting. For example, if: 

$$ W = \begin{bmatrix}
    j  & k  & l\\
    m  & n & o \\
    p  & q & r 
\end{bmatrix}\;\;\; X = \begin{bmatrix}
    a  & b  & c\\
    d  & e & f \\
    g  & h & i 
\end{bmatrix} \;\;\; b =\begin{bmatrix}
    s  \\
    t  \\
    u
\end{bmatrix}\tag{2}$$

Then $WX + b$ will be:

  
$$ WX + b = \begin{bmatrix}
    (ja + kd + lg) + s  & (jb + ke + lh) + s  & (jc + kf + li)+ s\\
    (ma + nd + og) + t & (mb + ne + oh) + t & (mc + nf + oi) + t\\
    (pa + qd + rg) + u & (pb + qe + rh) + u & (pc + qf + ri)+ u
\end{bmatrix}\tag{3}  $$

**Exercise**: Implement initialization for an L-layer Neural Network. 

**Instructions**:
- The model's structure is *[LINEAR -> RELU] $ \times$ (L-1) -> LINEAR -> SIGMOID*. I.e., it has $L-1$ layers using a ReLU activation function followed by an output layer with a sigmoid activation function.
- Use random initialization for the weight matrices. Use `np.random.randn(shape) * 0.01`.
- Use zeros initialization for the biases. Use `np.zeros(shape)`.
- We will store $n^{[l]}$, the number of units in different layers, in a variable `layer_dims`. For example, the `layer_dims` for the "Planar Data classification model" from last week would have been [2,4,1]: There were two inputs, one hidden layer with 4 hidden units, and an output layer with 1 output unit. Thus means `W1`'s shape was (4,2), `b1` was (4,1), `W2` was (1,4) and `b2` was (1,1). Now you will generalize this to $L$ layers! 
- Here is the implementation for $L=1$ (one layer neural network). It should inspire you to implement the general case (L-layer neural network).

    if L == 1:
        parameters["W" + str(L)] = np.random.randn(layer_dims[1], layer_dims[0]) * 0.01
        parameters["b" + str(L)] = np.zeros((layer_dims[1], 1))

-------------------------------------------------------

	# GRADED FUNCTION: initialize_parameters_deep
	
	def initialize_parameters_deep(layer_dims):
	    """
	    Arguments:
	    layer_dims -- python array (list) containing the dimensions of each layer in our network
	    
	    Returns:
	    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
	                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
	                    bl -- bias vector of shape (layer_dims[l], 1)
	    """
	    
	    np.random.seed(3)
	    parameters = {}
	    L = len(layer_dims)            # number of layers in the network
	
	    for l in range(1, L):
	        ### START CODE HERE ### (≈ 2 lines of code)
	        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
	        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
	        ### END CODE HERE ###
	        
	        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
	        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
	
	        
	    return parameters

	parameters = initialize_parameters_deep([5,4,3])
	print("W1 = " + str(parameters["W1"]))
	print("b1 = " + str(parameters["b1"]))
	print("W2 = " + str(parameters["W2"]))
	print("b2 = " + str(parameters["b2"]))

**Expected output**:
       
<table style="width:80%">
  <tr>
    <td> **W1** </td>
    <td>[[ 0.01788628  0.0043651   0.00096497 -0.01863493 -0.00277388]
 [-0.00354759 -0.00082741 -0.00627001 -0.00043818 -0.00477218]
 [-0.01313865  0.00884622  0.00881318  0.01709573  0.00050034]
 [-0.00404677 -0.0054536  -0.01546477  0.00982367 -0.01101068]]</td> 
  </tr>
  
  <tr>
    <td>**b1** </td>
    <td>[[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]</td> 
  </tr>
  
  <tr>
    <td>**W2** </td>
    <td>[[-0.01185047 -0.0020565   0.01486148  0.00236716]
 [-0.01023785 -0.00712993  0.00625245 -0.00160513]
 [-0.00768836 -0.00230031  0.00745056  0.01976111]]</td> 
  </tr>
  
  <tr>
    <td>**b2** </td>
    <td>[[ 0.]
 [ 0.]
 [ 0.]]</td> 
  </tr>
  
</table>

## 4 - Forward propagation module

### 4.1 - Linear Forward 
Now that you have initialized your parameters, you will do the forward propagation module. You will start by implementing some basic functions that you will use later when implementing the model. You will complete three functions in this order:

- LINEAR
- LINEAR -> ACTIVATION where ACTIVATION will be either ReLU or Sigmoid. 
- [LINEAR -> RELU] $\times$ (L-1) -> LINEAR -> SIGMOID (whole model)

The linear forward module (vectorized over all the examples) computes the following equations:

$$Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}\tag{4}$$

where $A^{[0]} = X$. 

**Exercise**: Build the linear part of forward propagation.

**Reminder**:
The mathematical representation of this unit is $Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}$. You may also find `np.dot()` useful. If your dimensions don't match, printing `W.shape` may help.


	# GRADED FUNCTION: linear_forward
	
	def linear_forward(A, W, b):
	    """
	    Implement the linear part of a layer's forward propagation.
	
	    Arguments:
	    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
	    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	    b -- bias vector, numpy array of shape (size of the current layer, 1)
	
	    Returns:
	    Z -- the input of the activation function, also called pre-activation parameter 
	    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
	    """
	    
	    ### START CODE HERE ### (≈ 1 line of code)
	    Z = np.dot(W,A) + b
	    ### END CODE HERE ###
	    
	    assert(Z.shape == (W.shape[0], A.shape[1]))
	    cache = (A, W, b)
	    
	    return Z, cache


	A, W, b = linear_forward_test_case()
	Z, linear_cache = linear_forward(A, W, b)
	print("Z = " + str(Z))

---------------------------

**Expected output**:

<table style="width:35%">
  
  <tr>
    <td> **Z** </td>
    <td> [[ 3.26295337 -1.23429987]] </td> 
  </tr>
  
</table>

### np.dot()  

对于二维矩阵，是计算矩阵的乘积

	A = np.array([[1,2,3],[4,5,6]])
	B = np.array([[1,2],[3,4],[5,6]])
	np.dot(A,B)
执行结果：
 
	array([[22, 28],
	       [49, 64]])

对于一维矩阵，则是计算两者的内积     

	C = np.array([1,2,3])
	D = np.array([4,5,6])
	np.dot(C,D)

执行结果：   

	32



### 4.2 - Linear-Activation Forward

In this notebook, you will use two activation functions:

- **Sigmoid**: $\sigma(Z) = \sigma(W A + b) = \frac{1}{ 1 + e^{-(W A + b)}}$. We have provided you with the `sigmoid` function. This function returns **two** items: the activation value "`a`" and a "`cache`" that contains "`Z`" (it's what we will feed in to the corresponding backward function). To use it you could just call: 
``` python
A, activation_cache = sigmoid(Z)
```

- **ReLU**: The mathematical formula for ReLu is $A = RELU(Z) = max(0, Z)$. We have provided you with the `relu` function. This function returns **two** items: the activation value "`A`" and a "`cache`" that contains "`Z`" (it's what we will feed in to the corresponding backward function). To use it you could just call:
``` python
A, activation_cache = relu(Z)
```

For more convenience, you are going to group two functions (Linear and Activation) into one function (LINEAR->ACTIVATION). Hence, you will implement a function that does the LINEAR forward step followed by an ACTIVATION forward step.

**Exercise**: Implement the forward propagation of the *LINEAR->ACTIVATION* layer. Mathematical relation is: $A^{[l]} = g(Z^{[l]}) = g(W^{[l]}A^{[l-1]} +b^{[l]})$ where the activation "g" can be sigmoid() or relu(). Use linear_forward() and the correct activation function.



	# GRADED FUNCTION: linear_activation_forward
	
	def linear_activation_forward(A_prev, W, b, activation):
	    """
	    Implement the forward propagation for the LINEAR->ACTIVATION layer
	
	    Arguments:
	    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
	    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	    b -- bias vector, numpy array of shape (size of the current layer, 1)
	    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
	
	    Returns:
	    A -- the output of the activation function, also called the post-activation value 
	    cache -- a python dictionary containing "linear_cache" and "activation_cache";
	             stored for computing the backward pass efficiently
	    """
	    
	    if activation == "sigmoid":
	        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
	        ### START CODE HERE ### (≈ 2 lines of code)
	        Z, linear_cache = linear_forward(A_prev, W, b)
	        A, activation_cache =  sigmoid(Z)
	        ### END CODE HERE ###
	    
	    elif activation == "relu":
	        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
	        ### START CODE HERE ### (≈ 2 lines of code)
	        Z, linear_cache = linear_forward(A_prev, W, b)
	        A, activation_cache = relu(Z)
	        ### END CODE HERE ###
	    
	    assert (A.shape == (W.shape[0], A_prev.shape[1]))
	    cache = (linear_cache, activation_cache)
	
	    return A, cache

-------------------------------------------------------

	A_prev, W, b = linear_activation_forward_test_case()
	
	A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
	print("With sigmoid: A = " + str(A))
	
	A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
	print("With ReLU: A = " + str(A))

----------------------
**Expected output**:
       
<table style="width:35%">
  <tr>
    <td> **With sigmoid: A ** </td>
    <td > [[ 0.96890023  0.11013289]]</td> 
  </tr>
  <tr>
    <td> **With ReLU: A ** </td>
    <td > [[ 3.43896131  0.        ]]</td> 
  </tr>
</table> 

**Note**: In deep learning, the "[LINEAR->ACTIVATION]" computation is counted as a single layer in the neural network, not two layers.   

### sigmoid() && relu() 

在这部分练习中，不需要去实现sigmoid()和relu()，直接调用即可。 如果在这里实现这两个函数，即使运行结果正确，也会无法通过测试。  

### d) L-Layer Model 

For even more convenience when implementing the $L$-layer Neural Net, you will need a function that replicates the previous one (`linear_activation_forward` with RELU) $L-1$ times, then follows that with one `linear_activation_forward` with SIGMOID. 

   ![](/images/images_2018/model_architecture_kiank.png)    

<center> **Figure 2** : *[LINEAR -> RELU] $\times$ (L-1) -> LINEAR -> SIGMOID* model</center>

**Exercise**: Implement the forward propagation of the above model.

**Instruction**: In the code below, the variable `AL` will denote $A^{[L]} = \sigma(Z^{[L]}) = \sigma(W^{[L]} A^{[L-1]} + b^{[L]})$. (This is sometimes also called `Yhat`, i.e., this is $\hat{Y}$.) 

**Tips**:
- Use the functions you had previously written 
- Use a for loop to replicate [LINEAR->RELU] (L-1) times
- Don't forget to keep track of the caches in the "caches" list. To add a new value `c` to a `list`, you can use `list.append(c)`.

	# GRADED FUNCTION: L_model_forward
	
	def L_model_forward(X, parameters):
	    """
	    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
	    
	    Arguments:
	    X -- data, numpy array of shape (input size, number of examples)
	    parameters -- output of initialize_parameters_deep()
	    
	    Returns:
	    AL -- last post-activation value
	    caches -- list of caches containing:
	                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
	    """
	
	    caches = []
	    A = X
	    L = len(parameters) // 2                  # number of layers in the neural network
	    
	    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
	    for l in range(1, L):
	        A_prev = A 
	        ### START CODE HERE ### (≈ 2 lines of code)
	        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
	        caches.append(cache)
	        ### END CODE HERE ###
	    
	    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
	    ### START CODE HERE ### (≈ 2 lines of code)
	    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
	    caches.append(cache)
	    ### END CODE HERE ###
	    
	    assert(AL.shape == (1,X.shape[1]))
	            
	    return AL, caches

-------------------------------

	X, parameters = L_model_forward_test_case_2hidden()
	AL, caches = L_model_forward(X, parameters)
	print("AL = " + str(AL))
	print("Length of caches list = " + str(len(caches)))

---------------------------------
<table style="width:50%">
  <tr>
    <td> **AL** </td>
    <td > [[ 0.03921668  0.70498921  0.19734387  0.04728177]]</td> 
  </tr>
  <tr>
    <td> **Length of caches list ** </td>
    <td > 3 </td> 
  </tr>
</table>

Great! Now you have a full forward propagation that takes the input X and outputs a row vector $A^{[L]}$ containing your predictions. It also records all intermediate values in "caches". Using $A^{[L]}$, you can compute the cost of your predictions.



## 5 - Cost function

Now you will implement forward and backward propagation. You need to compute the cost, because you want to check if your model is actually learning.

**Exercise**: Compute the cross-entropy cost $J$, using the following formula: $$-\frac{1}{m} \sum\limits_{i = 1}^{m} (y^{(i)}\log\left(a^{[L] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right)) \tag{7}$$


	# GRADED FUNCTION: compute_cost
	
	def compute_cost(AL, Y):
	    """
	    Implement the cost function defined by equation (7).
	
	    Arguments:
	    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
	    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
	
	    Returns:
	    cost -- cross-entropy cost
	    """
	    
	    m = Y.shape[1]
	
	    # Compute loss from aL and y.
	    ### START CODE HERE ### (≈ 1 lines of code)
	    cost = -1/m * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))
	    ### END CODE HERE ###
	    
	    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	    assert(cost.shape == ())
	    
	    return cost


-------------------------------------------

	Y, AL = compute_cost_test_case()
	
	print("cost = " + str(compute_cost(AL, Y)))

---------------------------------------------

**Expected Output**:

<table>

    <tr>
    <td>**cost** </td>
    <td> 0.41493159961539694</td> 
    </tr>
</table>

