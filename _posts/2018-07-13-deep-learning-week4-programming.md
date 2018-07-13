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

- Initialize the parameters for a two-layer network and for an $L$-layer neural network.
- Implement the forward propagation module (shown in purple in the figure below).
     - Complete the LINEAR part of a layer's forward propagation step (resulting in $Z^{[l]}$).
     - We give you the ACTIVATION function (relu/sigmoid).
     - Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
     - Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer $L$). This gives you a new L_model_forward function.
- Compute the loss.
- Implement the backward propagation module (denoted in red in the figure below).
    - Complete the LINEAR part of a layer's backward propagation step.
    - We give you the gradient of the ACTIVATE function (relu_backward/sigmoid_backward) 
    - Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function.
    - Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
- Finally update the parameters.  

   ![](/images/images_2018/final outline.png)      
       
<caption><center> **Figure 1**</center></caption><br>


**Note** that for every forward function, there is a corresponding backward function. That is why at every step of your forward module you will be storing some values in a cache. The cached values are useful for computing gradients. In the backpropagation module you will then use the cache to calculate the gradients. This assignment will show you exactly how to carry out each of these steps. 

## 3 - Initialization

You will write two helper functions that will initialize the parameters for your model. The first function will be used to initialize parameters for a two layer model. The second one will generalize this initialization process to $L$ layers.

### 3.1 - 2-layer Neural Network

**Exercise**: Create and initialize the parameters of the 2-layer neural network.

**Instructions**:
- The model's structure is: *LINEAR -> RELU -> LINEAR -> SIGMOID*. 
- Use random initialization for the weight matrices. Use `np.random.randn(shape)*0.01` with the correct shape.
- Use zero initialization for the biases. Use `np.zeros(shape)`.
     


    

   ![](/images/images_2018/7-3_36.png)  




 
     

