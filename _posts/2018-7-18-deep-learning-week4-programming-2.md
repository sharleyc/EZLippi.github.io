---
layout:     post
title:      深层神经网络编程作业(下)  
keywords:   博客
categories: [机器学习]
tags:	    [深度学习，深层神经网络]
---

这部分是深层神经网络的编程作业的下篇，主要是后向传播的实现。 


## 6 - Backward propagation module

Just like with forward propagation, you will implement helper functions for backpropagation. Remember that back propagation is used to calculate the gradient of the loss function with respect to the parameters. 

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

Now, similar to forward propagation, you are going to build the backward propagation in three steps:   
- LINEAR backward        
- LINEAR -> ACTIVATION backward where ACTIVATION computes the derivative of either the ReLU or sigmoid activation   
- [LINEAR -> RELU] $\times$ (L-1) -> LINEAR -> SIGMOID backward (whole model)


### 6.1 - Linear backward

For layer $l$, the linear part is: $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$ (followed by an activation).

Suppose you have already calculated the derivative $dZ^{[l]} = \frac{\partial \mathcal{L} }{\partial Z^{[l]}}$. You want to get $(dW^{[l]}, db^{[l]} dA^{[l-1]})$.

  ![](/images/images_2018/linearback_kiank.png)  

<center> **Figure 4** </center>

The three outputs $(dW^{[l]}, db^{[l]}, dA^{[l]})$ are computed using the input $dZ^{[l]}$.Here are the formulas you need:   
$$ dW^{[l]} = \frac{\partial \mathcal{L} }{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} A^{[l-1] T} \tag{8}$$   
$$ db^{[l]} = \frac{\partial \mathcal{L} }{\partial b^{[l]}} = \frac{1}{m} \sum_{i = 1}^{m} dZ^{[l](i)}\tag{9}$$   
$$ dA^{[l-1]} = \frac{\partial \mathcal{L} }{\partial A^{[l-1]}} = W^{[l] T} dZ^{[l]} \tag{10}$$  


**Exercise**: Use the 3 formulas above to implement linear_backward().  

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
    <td> **dA_prev** </td>
    <td > [[ 0.51822968 -0.19517421]
 [-0.40506361  0.15255393]
 [ 2.37496825 -0.89445391]] </td> 
  </tr> 
  
    <tr>
        <td> **dW** </td>
        <td > [[-0.10076895  1.40685096  1.64992505]] </td> 
    </tr> 
  
    <tr>
        <td> **db** </td>
        <td> [[ 0.50629448]] </td> 
    </tr> 
    
</table>


### 6.2 - Linear-Activation backward

Next, you will create a function that merges the two helper functions: **`linear_backward`** and the backward step for the activation **`linear_activation_backward`**. 

To help you implement `linear_activation_backward`, we provided two backward functions:     

- **`sigmoid_backward`**: Implements the backward propagation for SIGMOID unit. You can call it as follows:

```
dZ = sigmoid_backward(dA, activation_cache)
```

- **`relu_backward`**: Implements the backward propagation for RELU unit. You can call it as follows:

```
dZ = relu_backward(dA, activation_cache)
```

If $g(.)$ is the activation function, 
`sigmoid_backward` and `relu_backward` compute $$dZ^{[l]} = dA^{[l]} * g'(Z^{[l]}) \tag{11}$$.  

**Exercise**: Implement the backpropagation for the *LINEAR->ACTIVATION* layer. 


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


### 6.3 - L-Model Backward 

Now you will implement the backward function for the whole network. Recall that when you implemented the `L_model_forward` function, at each iteration, you stored a cache which contains (X,W,b, and z). In the back propagation module, you will use those variables to compute the gradients. Therefore, in the `L_model_backward` function, you will iterate through all the hidden layers backward, starting from layer $L$. On each step, you will use the cached values for layer $l$ to backpropagate through layer $l$. Figure 5 below shows the backward pass. 

  ![](/images/images_2018/mn_backward.png)   

<center>  **Figure 5** : Backward pass  </center>

** Initializing backpropagation**:
To backpropagate through this network, we know that the output is, 
$A^{[L]} = \sigma(Z^{[L]})$. Your code thus needs to compute `dAL` $= \frac{\partial \mathcal{L}}{\partial A^{[L]}}$.
To do so, use this formula (derived using calculus which you don't need in-depth knowledge of):
```python
dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
```

You can then use this post-activation gradient `dAL` to keep going backward. As seen in Figure 5, you can now feed in `dAL` into the LINEAR->SIGMOID backward function you implemented (which will use the cached values stored by the L_model_forward function). After that, you will have to use a `for` loop to iterate through all the other layers using the LINEAR->RELU backward function. You should store each dA, dW, and db in the grads dictionary. To do so, use this formula : 

$$grads["dW" + str(l)] = dW^{[l]}\tag{15} $$

For example, for $l=3$ this would store $dW^{[l]}$ in `grads["dW3"]`.

**Exercise**: Implement backpropagation for the *[LINEAR->RELU] $\times$ (L-1) -> LINEAR -> SIGMOID* model. 



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

### 6.4 - Update Parameters

In this section you will update the parameters of the model, using gradient descent: 

$$ W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]} \tag{16}$$
$$ b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]} \tag{17}$$

where $\alpha$ is the learning rate. After computing the updated parameters, store them in the parameters dictionary. 


**Exercise**: Implement `update_parameters()` to update your parameters using gradient descent.

**Instructions**:
Update parameters using gradient descent on every $W^{[l]}$ and $b^{[l]}$ for $l = 1, 2, ..., L$.  

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
	        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW"+str(l+1)]
	        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db"+str(l+1)]
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

## 7 - Conclusion

Congrats on implementing all the functions required for building a deep neural network! 

We know it was a long assignment but going forward it will only get better. The next part of the assignment is easier. 

In the next assignment you will put all these together to build two models:
- A two-layer neural network
- An L-layer neural network

You will in fact use these models to classify cat vs non-cat images!