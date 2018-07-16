---
layout:     post
title:      深度学习之深层神经网络--编程作业
keywords:   博客
categories: [机器学习]
tags:	    [深度学习，深层神经网络]
---

## 1 - Packages

Let's first import all the packages that you will need during this assignment.          

-  [numpy](www.numpy.org) is the main package for scientific computing with Python.
-  [matplotlib](http://matplotlib.org) is a library to plot graphs in Python.   
- dnn_utils provides some necessary functions for this notebook.   
- testCases provides some test cases to assess the correctness of your functions
- np.random.seed(1) is used to keep all the random function calls consistent. It will help us grade your work. Please don't change the seed. 


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

## 2 - Outline of the Assignment

To build your neural network, you will be implementing several "helper functions". These helper functions will be used in the next assignment to build a two-layer neural network and an L-layer neural network. Each small helper function you will implement will have detailed instructions that will walk you through the necessary steps. Here is an outline of this assignment, you will:

- Initialize the parameters for a two-layer network and for an **L**-layer neural network.
- Implement the forward propagation module (shown in purple in the figure below).
     - Complete the LINEAR part of a layer's forward propagation step (resulting in **Z**<sup>[l]</sup>).
     - We give you the ACTIVATION function (relu/sigmoid).
     - Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
     - Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer **L**). This gives you a new L_model_forward function.
- Compute the loss.
- Implement the backward propagation module (denoted in red in the figure below).
    - Complet the LINEAR part of a layer's backward propagation step.
    - We give you the gradient of the ACTIVATE function (relu_backward/sigmoid_backward) 
    - Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function.
    - Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
- Finally update the parameters.  

   ![](/images/images_2018/7-13_01.png)         
       
 <center>**Figure 1**</center> 


**Note** that for every forward function, there is a corresponding backward function. That is why at every step of your forward module you will be storing some values in a cache. The cached values are useful for computing gradients. In the backpropagation module you will then use the cache to calculate the gradients. This assignment will show you exactly how to carry out each of these steps. 

## 3 - Initialization

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
```python
    if L == 1:
        parameters["W" + str(L)] = np.random.randn(layer_dims[1], layer_dims[0]) * 0.01
        parameters["b" + str(L)] = np.zeros((layer_dims[1], 1))

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