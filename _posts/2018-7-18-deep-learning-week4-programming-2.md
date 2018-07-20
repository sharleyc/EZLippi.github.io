---
layout:     post
title:      深度神经网络编程作业(二)  
keywords:   博客
categories: [机器学习]
tags:	    [深度学习，深层神经网络]
---

这部分是深层神经网络的编程作业的下篇，主要是反向传播的实现。 


## 6 - 反向传播模块 

同前向传播类似，你需要实现一些辅助函数。请记住反向传播是用来计算损失函数关于某些参数的梯度。

**Reminder**:      

![](/images/images_2018/backprop_kiank.png)  

<center> **Figure 3** : Forward and Backward propagation for *LINEAR->RELU->LINEAR->SIGMOID* <br> *The purple blocks represent the forward propagation, and the red blocks represent the backward propagation.*  </center>

<!-- 
For those of you who are expert in calculus (you don't need to be to do this assignment), the chain rule of calculus can be used to derive the derivative of the loss $\mathcal{L}$ with respect to $z^{[1]}$ in a 2-layer network as follows:

$$\frac{d \mathcal{L}(a^{[2]},y)}{{dz^{[1]}}} = \frac{d\mathcal{L}(a^{[2]},y)}{{da^{[2]}}}\frac{{da^{[2]}}}{{dz^{[2]}}}\frac{{dz^{[2]}}}{{da^{[1]}}}\frac{{da^{[1]}}}{{dz^{[1]}}} \tag{8} $$

In order to calculate the gradient $dW^{[1]} = \frac{\partial L}{\partial W^{[1]}}$, you use the previous chain rule and you do $dW^{[1]} = dz^{[1]} \times \frac{\partial z^{[1]} }{\partial W^{[1]}}$. During the backpropagation, at each step you multiply your current gradient by the gradient corresponding to the specific layer to get the gradient you wanted.

Equivalently, in order to calculate the gradient $db^{[1]} = \frac{\partial L}{\partial b^{[1]}}$, you use the previous chain rule and you do $db^{[1]} = dz^{[1]} \times \frac{\partial z^{[1]} }{\partial b^{[1]}}$.

This is why we talk about **backpropagation**.
!-->

类似前向传播，你将用三步来完成反向传播： 
  
- LINEAR 反向传播        
- LINEAR -> ACTIVATION 反向传播，计算激励函数的偏导数
- [LINEAR -> RELU] $\times$ (L-1) -> LINEAR -> SIGMOID 反向传播 (整个模型)


### 6.1 - 线性反向传播

层 $l$, 线性公式是： $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$ 。

假设你已经计算出偏导数： $dZ^{[l]} = \frac{\partial \mathcal{L} }{\partial Z^{[l]}}$。 你想计算出 $(dW^{[l]}, db^{[l]} dA^{[l-1]})$。

  ![](/images/images_2018/linearback_kiank.png)  

<center> **Figure 4** </center>

利用输入 $dZ^{[l]}$ 可以计算出3个输出 $(dW^{[l]}, db^{[l]}, dA^{[l]})。你需要用到以下三个公式：       
   
$$ dW^{[l]} = \frac{\partial \mathcal{L} }{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} A^{[l-1] T} \tag{8}$$   
$$ db^{[l]} = \frac{\partial \mathcal{L} }{\partial b^{[l]}} = \frac{1}{m} \sum_{i = 1}^{m} dZ^{[l](i)}\tag{9}$$   
$$ dA^{[l-1]} = \frac{\partial \mathcal{L} }{\partial A^{[l-1]}} = W^{[l] T} dZ^{[l]} \tag{10}$$  


**Exercise**: 利用以上3个公式完成函数linear_backward()的编写。   

	# GRADED FUNCTION: linear_backward
	
	def linear_backward(dZ, cache):
	    """
	    Implement the linear portion of backward propagation for a single layer (layer l)
	
	    Arguments:
	    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
	    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
	
	    Returns:
	    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	    db -- Gradient of the cost with respect to b (current layer l), same shape as b
	    """
	    A_prev, W, b = cache
	    m = A_prev.shape[1]
	    ### START CODE HERE ### (≈ 3 lines of code)
	    dW = 1 / m * np.dot(dZ, A_prev.T)
	    db = 1 / m * np.sum(dZ, axis = 1, keepdims = True)
	    dA_prev = np.dot(W.T, dZ) # W.T * dZ is wrong 
	    ### END CODE HERE ###
	    
	    assert (dA_prev.shape == A_prev.shape)
	    assert (dW.shape == W.shape)
	    assert (db.shape == b.shape)
	    
	    return dA_prev, dW, db

--------------------------------


	# Set up some test inputs
	dZ, linear_cache = linear_backward_test_case()
	
	dA_prev, dW, db = linear_backward(dZ, linear_cache)
	print ("dA_prev = "+ str(dA_prev))
	print ("dW = " + str(dW))
	print ("db = " + str(db))

---------------------------------------


**Expected Output**: 

<table style="width:90%">
  <tr>
    <td> dA_prev </td>
    <td > [[ 0.51822968 -0.19517421]
 [-0.40506361  0.15255393]
 [ 2.37496825 -0.89445391]] </td> 
  </tr> 
  
    <tr>
        <td> dW </td>
        <td > [[-0.10076895  1.40685096  1.64992505]] </td> 
    </tr> 
  
    <tr>
        <td> db </td>
        <td> [[ 0.50629448]] </td> 
    </tr> 
    
</table>


### 6.2 - 线性激活函数反向传播

接下来是完成**`linear_activation_backward`**函数的编写，这个函数合并了两个辅助函数：**`linear_backward`**，以及激励单元的反向传播。

我们提供了两个反向传播函数，来帮助你完成`linear_activation_backward`。    

- **`sigmoid_backward`**: 完成SIGMOID单元的反向传播。调用方法如下：   

```
dZ = sigmoid_backward(dA, activation_cache)
```

- **`relu_backward`**: 完成RELU单元的反向传播。调用方法如下：

```
dZ = relu_backward(dA, activation_cache)
```

假设 $g(.)$ 是激励函数， 
`sigmoid_backward` 和 `relu_backward` 计算的是 $$dZ^{[l]} = dA^{[l]} * g'(Z^{[l]}) \tag{11}$$。

**Exercise**: 完成 *LINEAR->ACTIVATION* 部分的反向传播算法。 


	# GRADED FUNCTION: linear_activation_backward
	
	def linear_activation_backward(dA, cache, activation):
	    """
	    Implement the backward propagation for the LINEAR->ACTIVATION layer.
	    
	    Arguments:
	    dA -- post-activation gradient for current layer l 
	    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
	    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
	    
	    Returns:
	    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	    db -- Gradient of the cost with respect to b (current layer l), same shape as b
	    """
	    linear_cache, activation_cache = cache
	    
	    if activation == "relu":
	        ### START CODE HERE ### (≈ 2 lines of code)
	        dZ = relu_backward(dA, activation_cache) 
	        dA_prev, dW, db = linear_backward(dZ, linear_cache)
	        ### END CODE HERE ###
	        
	    elif activation == "sigmoid":
	        ### START CODE HERE ### (≈ 2 lines of code)
	        dZ = sigmoid_backward(dA, activation_cache)
	        dA_prev, dW, db = linear_backward(dZ, linear_cache)
	        ### END CODE HERE ###
	    
	    return dA_prev, dW, db

---------------------------------------------------------

	dAL, linear_activation_cache = linear_activation_backward_test_case()
	
	dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
	print ("sigmoid:")
	print ("dA_prev = "+ str(dA_prev))
	print ("dW = " + str(dW))
	print ("db = " + str(db) + "\n")
	
	dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
	print ("relu:")
	print ("dA_prev = "+ str(dA_prev))
	print ("dW = " + str(dW))
	print ("db = " + str(db))

--------------------------------------------------------

**Expected output with sigmoid:**

<table style="width:100%">
  <tr>
    <td > dA_prev </td> 
           <td >[[ 0.11017994  0.01105339]
 [ 0.09466817  0.00949723]
 [-0.05743092 -0.00576154]] </td> 

  </tr> 
  
    <tr>
    <td > dW </td> 
           <td > [[ 0.10266786  0.09778551 -0.01968084]] </td> 
  </tr> 
  
    <tr>
    <td > db </td> 
           <td > [[-0.05729622]] </td> 
  </tr> 
</table>


**Expected output with relu:**

<table style="width:100%">
  <tr>
    <td > dA_prev </td> 
           <td > [[ 0.44090989  0.        ]
 [ 0.37883606  0.        ]
 [-0.2298228   0.        ]] </td> 

  </tr> 
  
    <tr>
    <td > dW </td> 
           <td > [[ 0.44513824  0.37371418 -0.10478989]] </td> 
  </tr> 
  
    <tr>
    <td > db </td> 
           <td > [[-0.20837892]] </td> 
  </tr> 
</table>


### 6.3 - L层模型 反向传播

你将实现整个神经网络的反向传播函数。回顾一下，你在之前的`L_model_forward`函数中，存储了(X,W,b, and z)。在反向传播模块中，你将会用到这些变量来计算梯度下降。因此在`L_model_backward`函数中，你将从**L**层开始迭代反向传播算法：  
 

  ![](/images/images_2018/mn_backward.png)   

<center>  **Figure 5** : Backward pass  </center>

** 反向传播的初始化**： 

我们可以利用以下公式来计算AL的偏导数：

```
dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
```

接下来你就可以利用`dAL`来进行反向传播了。将`dAL` 和对应的缓存变量带入LINEAR->SIGMOID 反向传播函数中，得到dAL-1,dWL,dbL，在循环迭代LINEAR->RELU 反向传播函数，将得到的dA,dW,db存放在grads字典中：   


$$grads["dW" + str(l)] = dW^{[l]}\tag{15} $$

比如，如果 $l=3$ 那么 $dW^{[l]}$ 存储在 `grads["dW3"]`中。  

**Exercise**: 完成*[LINEAR->RELU] $\times$ (L-1) -> LINEAR -> SIGMOID* 部分的反向传播算法。



	# GRADED FUNCTION: L_model_backward
	
	def L_model_backward(AL, Y, caches):
	    """
	    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
	    
	    Arguments:
	    AL -- probability vector, output of the forward propagation (L_model_forward())
	    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
	    caches -- list of caches containing:
	                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
	                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
	    
	    Returns:
	    grads -- A dictionary with the gradients
	             grads["dA" + str(l)] = ... 
	             grads["dW" + str(l)] = ...
	             grads["db" + str(l)] = ... 
	    """
	    grads = {}
	    L = len(caches) # the number of layers
	    m = AL.shape[1]
	    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
	    
	    # Initializing the backpropagation
	    ### START CODE HERE ### (1 line of code)
	    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
	    ### END CODE HERE ###
	    
	    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
	    ### START CODE HERE ### (approx. 2 lines)
	    current_cache = caches[L-1]
	    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
	    ### END CODE HERE ###
	    
	    # Loop from l=L-2 to l=0
	    for l in reversed(range(L-1)):
	        # lth layer: (RELU -> LINEAR) gradients.
	        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
	        ### START CODE HERE ### (approx. 5 lines)
	        current_cache = caches[l]
	        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
	        grads["dA" + str(l)] = dA_prev_temp
	        grads["dW" + str(l + 1)] = dW_temp
	        grads["db" + str(l + 1)] = db_temp
	        ### END CODE HERE ###
	
	    return grads
	
-----------------------------------------------

	AL, Y_assess, caches = L_model_backward_test_case()
	grads = L_model_backward(AL, Y_assess, caches)
	print_grads(grads)

------------------------------------------------

**Expected Output**

<table style="width:60%">
  
  <tr>
    <td > dW1 </td> 
           <td > [[ 0.41010002  0.07807203  0.13798444  0.10502167]
 [ 0.          0.          0.          0.        ]
 [ 0.05283652  0.01005865  0.01777766  0.0135308 ]] </td> 
  </tr> 
  
    <tr>
    <td > db1 </td> 
           <td > [[-0.22007063]
 [ 0.        ]
 [-0.02835349]] </td> 
  </tr> 
  
  <tr>
  <td > dA1 </td> 
           <td > [[ 0.12913162 -0.44014127]
 [-0.14175655  0.48317296]
 [ 0.01663708 -0.05670698]] </td> 

  </tr> 
</table>

### 6.4 - 更新参数

在这个部分，你将使用梯度下降算法来更新模型的参数：    

$$ W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]} \tag{16}$$
$$ b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]} \tag{17}$$

其中 $\alpha$ 是学习速率。计算出新参数后，将它们存储在parameters 字典中。  


**Exercise**: 实现 `update_parameters()` 函数。 

**Instructions**:
使用梯度下降算法，更新计算每一个参数  
 $W^{[l]}$ and $b^{[l]}$ for $l = 1, 2, ..., L$.  

	# GRADED FUNCTION: update_parameters
	
	def update_parameters(parameters, grads, learning_rate):
	    """
	    Update parameters using gradient descent
	    
	    Arguments:
	    parameters -- python dictionary containing your parameters 
	    grads -- python dictionary containing your gradients, output of L_model_backward
	    
	    Returns:
	    parameters -- python dictionary containing your updated parameters 
	                  parameters["W" + str(l)] = ... 
	                  parameters["b" + str(l)] = ...
	    """
	    
	    L = len(parameters) // 2 # number of layers in the neural network
	
	    # Update rule for each parameter. Use a for loop.
	    ### START CODE HERE ### (≈ 3 lines of code)
	    for l in range(L):
	        parameters["W" + str(l+1)] -= learning_rate * grads["dW"+str(l+1)]
	        parameters["b" + str(l+1)] -= learning_rate * grads["db"+str(l+1)]
	    ### END CODE HERE ###
	    return parameters 

---------------------------------------------------

	parameters, grads = update_parameters_test_case()
	parameters = update_parameters(parameters, grads, 0.1)
	
	print ("W1 = "+ str(parameters["W1"]))
	print ("b1 = "+ str(parameters["b1"]))
	print ("W2 = "+ str(parameters["W2"]))
	print ("b2 = "+ str(parameters["b2"])) 

----------------------------------------------------

**Expected Output**:

<table style="width:100%"> 
    <tr>
    <td > W1 </td> 
           <td > [[-0.59562069 -0.09991781 -2.14584584  1.82662008]
 [-1.76569676 -0.80627147  0.51115557 -1.18258802]
 [-1.0535704  -0.86128581  0.68284052  2.20374577]] </td> 
  </tr> 
  
    <tr>
    <td > b1 </td> 
           <td > [[-0.04659241]
 [-1.28888275]
 [ 0.53405496]] </td> 
  </tr> 
  <tr>
    <td > W2 </td> 
           <td > [[-0.55569196  0.0354055   1.32964895]]</td> 
  </tr> 
  
    <tr>
    <td > b2 </td> 
           <td > [[-0.84610769]] </td> 
  </tr> 
</table>

----------------------------------

## 7 - 总结


恭喜你实现了构建深度神经网络所需的所有函数！
 
接下来的作业中，你需要完成两个模型：  

- 一个两层的神经网络模型
- 一个**L**层的神经网络模型

你将应用这些模型来区分给定的图片，哪些是猫，哪些不是。  