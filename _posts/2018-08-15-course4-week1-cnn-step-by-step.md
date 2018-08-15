---
layout:     post
title:      卷积神经网络-一步步构建CNN(编程)
keywords:   博客
categories: [机器学习]
tags:	    [卷积神经网络，ConvNets]
---



# Convolutional Neural Networks: Step by Step

Welcome to Course 4's first assignment! In this assignment, you will implement convolutional (CONV) and pooling (POOL) layers in numpy, including both forward propagation and (optionally) backward propagation. 

## 1 - Packages

Let's first import all the packages that you will need during this assignment. 
- [numpy](www.numpy.org) is the fundamental package for scientific computing with Python.
- [matplotlib](http://matplotlib.org) is a library to plot graphs in Python.
- np.random.seed(1) is used to keep all the random function calls consistent. It will help us grade your work.

	%matplotlib inline
	plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
	plt.rcParams['image.interpolation'] = 'nearest'
	plt.rcParams['image.cmap'] = 'gray'
	
	%load_ext autoreload
	%autoreload 2
	
	np.random.seed(1)



## 2 - Outline of the Assignment

You will be implementing the building blocks of a convolutional neural network! Each function you will implement will have detailed instructions that will walk you through the steps needed:

- Convolution functions, including:
    - Zero Padding
    - Convolve window 
    - Convolution forward
    - Convolution backward (optional)
- Pooling functions, including:
    - Pooling forward
    - Create mask 
    - Distribute value
    - Pooling backward (optional)
    
This notebook will ask you to implement these functions from scratch in `numpy`. In the next notebook, you will use the TensorFlow equivalents of these functions to build the following model:

   ![](/images/images_2018/model.png) 

**Note** that for every forward function, there is its corresponding backward equivalent. Hence, at every step of your forward module you will store some parameters in a cache. These parameters are used to compute gradients during backpropagation. 



## 3 - Convolutional Neural Networks

Although programming frameworks make convolutions easy to use, they remain one of the hardest concepts to understand in Deep Learning. A convolution layer transforms an input volume into an output volume of different size, as shown below. 


   ![](/images/images_2018/conv_nn.png) 

In this part, you will build every step of the convolution layer. You will first implement two helper functions: one for zero padding and the other for computing the convolution function itself. 

### 3.1 - Zero-Padding

Zero-padding adds zeros around the border of an image:

   ![](/images/images_2018/PAD.png) 

The main benefits of padding are the following:

- It allows you to use a CONV layer without necessarily shrinking the height and width of the volumes. This is important for building deeper networks, since otherwise the height/width would shrink as you go to deeper layers. An important special case is the "same" convolution, in which the height/width is exactly preserved after one layer. 

- It helps us keep more of the information at the border of an image. Without padding, very few values at the next layer would be affected by pixels as the edges of an image.

**Exercise**: Implement the following function, which pads all the images of a batch of examples X with zeros. [Use np.pad](https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html). Note if you want to pad the array "a" of shape $(5,5,5,5,5)$ with `pad = 1` for the 2nd dimension, `pad = 3` for the 4th dimension and `pad = 0` for the rest, you would do:
```
a = np.pad(a, ((0,0), (1,1), (0,0), (3,3), (0,0)), 'constant', constant_values = (..,..))
```

	# GRADED FUNCTION: zero_pad
	
	def zero_pad(X, pad):
	    """
	    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
	    as illustrated in Figure 1.
	    
	    Argument:
	    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
	    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
	    
	    Returns:
	    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
	    """
	    
	    ### START CODE HERE ### (≈ 1 line)
	    X_pad = np.pad(X,((0, 0),(pad, pad),(pad, pad),(0, 0)),"constant",constant_values=(0,0))
	    ### END CODE HERE ###
	    
	    return X_pad

    #test code
	np.random.seed(1)
	x = np.random.randn(4, 3, 3, 2)
	x_pad = zero_pad(x, 2)
	print ("x.shape =", x.shape)
	print ("x_pad.shape =", x_pad.shape)
	print ("x[1,1] =", x[1,1])
	print ("x_pad[1,1] =", x_pad[1,1])
	
	fig, axarr = plt.subplots(1, 2)
	axarr[0].set_title('x')
	axarr[0].imshow(x[0,:,:,0])
	axarr[1].set_title('x_pad')
	axarr[1].imshow(x_pad[0,:,:,0])


执行结果：

   ![](/images/images_2018/8-15_01.png) 

### 3.2 - Single step of convolution 

In this part, implement a single step of convolution, in which you apply the filter to a single position of the input. This will be used to build a convolutional unit, which: 

- Takes an input volume 
- Applies a filter at every position of the input
- Outputs another volume (usually of different size)


In a computer vision application, each value in the matrix on the left corresponds to a single pixel value, and we convolve a 3x3 filter with the image by multiplying its values element-wise with the original matrix, then summing them up and adding a bias. In this first step of the exercise, you will implement a single step of convolution, corresponding to applying a filter to just one of the positions to get a single real-valued output.

Later in this notebook, you'll apply this function to multiple positions of the input to implement the full convolutional operation.

**Exercise**: Implement conv_single_step(). [Hint](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.sum.html).

	# GRADED FUNCTION: conv_single_step
	
	def conv_single_step(a_slice_prev, W, b):
	    """
	    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
	    of the previous layer.
	    
	    Arguments:
	    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
	    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
	    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
	    
	    Returns:
	    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
	    """
	
	    ### START CODE HERE ### (≈ 2 lines of code)
	    # Element-wise product between a_slice and W. Do not add the bias yet.
	    s = a_slice_prev * W
	    # Sum over all entries of the volume s.
	    Z = sum(sum(sum(s)))
	    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
	    Z = Z + float(b)
	    ### END CODE HERE ###
	
	    return Z

    #test code
	np.random.seed(1)
	a_slice_prev = np.random.randn(4, 4, 3)
	W = np.random.randn(4, 4, 3)
	b = np.random.randn(1, 1, 1)
	
	Z = conv_single_step(a_slice_prev, W, b)
	print("Z =", Z)

执行结果：

Z = -6.99908945068

### 3.3 - Convolutional Neural Networks - Forward pass

In the forward pass, you will take many filters and convolve them on the input. Each 'convolution' gives you a 2D matrix output. You will then stack these outputs to get a 3D volume: 

**Exercise**: Implement the function below to convolve the filters W on an input activation A_prev. This function takes as input A_prev, the activations output by the previous layer (for a batch of m inputs), F filters/weights denoted by W, and a bias vector denoted by b, where each filter has its own (single) bias. Finally you also have access to the hyperparameters dictionary which contains the stride and the padding.



**Hint**: 
1. To select a 2x2 slice at the upper left corner of a matrix "a_prev" (shape (5,5,3)), you would do:
```python
a_slice_prev = a_prev[0:2,0:2,:]
```
This will be useful when you will define `a_slice_prev` below, using the `start/end` indexes you will define.   
2. To define a_slice you will need to first define its corners `vert_start`, `vert_end`, `horiz_start` and `horiz_end`. This figure may be helpful for you to find how each of the corner can be defined using h, w, f and s in the code below.


   ![](/images/images_2018/vert_horiz_kiank.png) 


**Reminder**:
The formulas relating the output shape of the convolution to the input shape is:
$$ n_H = \lfloor \frac{n_{H_{prev}} - f + 2 \times pad}{stride} \rfloor +1 $$
$$ n_W = \lfloor \frac{n_{W_{prev}} - f + 2 \times pad}{stride} \rfloor +1 $$
$$ n_C = \text{number of filters used in the convolution}$$

For this exercise, we won't worry about vectorization, and will just implement everything with for-loops.

	# GRADED FUNCTION: conv_forward
	
	def conv_forward(A_prev, W, b, hparameters):
	    """
	    Implements the forward propagation for a convolution function
	    
	    Arguments:
	    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
	    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
	    b -- Biases, numpy array of shape (1, 1, 1, n_C)
	    hparameters -- python dictionary containing "stride" and "pad"
	        
	    Returns:
	    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
	    cache -- cache of values needed for the conv_backward() function
	    """
	    
	    ### START CODE HERE ###
	    # Retrieve dimensions from A_prev's shape (≈1 line)  
	    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
	    
	    # Retrieve dimensions from W's shape (≈1 line)
	    (f, f, n_C_prev, n_C) = W.shape
	    
	    # Retrieve information from "hparameters" (≈2 lines)
	    stride = hparameters["stride"]
	    pad = hparameters["pad"]
	    
	    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
	    n_H = np.int((n_H_prev - f + 2 * pad)/stride) + 1
	    n_W = np.int((n_W_prev - f + 2 * pad)/stride) + 1
	    
	    # Initialize the output volume Z with zeros. (≈1 line)
	    Z = np.zeros((m,n_H,n_W,n_C))
	    
	    # Create A_prev_pad by padding A_prev
	    A_prev_pad = zero_pad(A_prev, pad)
	    
	    for i in range(m):                               # loop over the batch of training examples
	        a_prev_pad = A_prev_pad[i,:,:,:]                             # Select ith training example's padded activation
	        for h in range(n_H):                           # loop over vertical axis of the output volume
	            for w in range(n_W):                       # loop over horizontal axis of the output volume
	                for c in range(n_C):                   # loop over channels (= #filters) of the output volume
	                    
	                    # Find the corners of the current "slice" (≈4 lines)
	                    vert_start = h  * stride
	                    vert_end = vert_start + f
	                    horiz_start = w * stride
	                    horiz_end = horiz_start + f
	                    
	                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
	                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
	                    
	                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
	                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])
	                                        
	    ### END CODE HERE ###
	    
	    # Making sure your output shape is correct
	    assert(Z.shape == (m, n_H, n_W, n_C))
	    
	    # Save information in "cache" for the backprop
	    cache = (A_prev, W, b, hparameters)
	    
	    return Z, cache

    #test code
	np.random.seed(1)
	A_prev = np.random.randn(10,4,4,3)
	W = np.random.randn(2,2,3,8)
	b = np.random.randn(1,1,1,8)
	hparameters = {"pad" : 2,
	               "stride": 2}
	
	Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
	print("Z's mean =", np.mean(Z))
	print("Z[3,2,1] =", Z[3,2,1])
	print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])

执行结果：


Z's mean = 0.0489952035289    
Z[3,2,1] = [-0.61490741 -6.7439236  -2.55153897  1.75698377  3.56208902  0.53036437
  5.18531798  8.75898442]   
cache_conv[0][1][2][3] = [-0.20075807  0.18656139  0.41005165]

Finally, CONV layer should also contain an activation, in which case we would add the following line of code:


	# Convolve the window to get back one output neuron
	Z[i, h, w, c] = ...
	# Apply activation
	A[i, h, w, c] = activation(Z[i, h, w, c])


You don't need to do it here.

## 4 - Pooling layer 

The pooling (POOL) layer reduces the height and width of the input. It helps reduce computation, as well as helps make feature detectors more invariant to its position in the input. The two types of pooling layers are: 

- Max-pooling layer: slides an ($f, f$) window over the input and stores the max value of the window in the output.

- Average-pooling layer: slides an ($f, f$) window over the input and stores the average value of the window in the output.


   ![](/images/images_2018/max_pool1.png) 
   ![](/images/images_2018/a_pool.png) 


These pooling layers have no parameters for backpropagation to train. However, they have hyperparameters such as the window size $f$. This specifies the height and width of the fxf window you would compute a max or average over. 

### 4.1 - Forward Pooling
Now, you are going to implement MAX-POOL and AVG-POOL, in the same function. 

**Exercise**: Implement the forward pass of the pooling layer. Follow the hints in the comments below.

**Reminder**:
As there's no padding, the formulas binding the output shape of the pooling to the input shape is:
$$ n_H = \lfloor \frac{n_{H_{prev}} - f}{stride} \rfloor +1 $$
$$ n_W = \lfloor \frac{n_{W_{prev}} - f}{stride} \rfloor +1 $$
$$ n_C = n_{C_{prev}}$$

	def pool_forward(A_prev, hparameters, mode = "max"):
	    """
	    Implements the forward pass of the pooling layer
	    
	    Arguments:
	    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
	    hparameters -- python dictionary containing "f" and "stride"
	    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
	    
	    Returns:
	    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
	    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
	    """
	    
	    # Retrieve dimensions from the input shape
	    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
	    
	    # Retrieve hyperparameters from "hparameters"
	    f = hparameters["f"]
	    stride = hparameters["stride"]
	    
	    # Define the dimensions of the output
	    n_H = int(1 + (n_H_prev - f) / stride)
	    n_W = int(1 + (n_W_prev - f) / stride)
	    n_C = n_C_prev
	    
	    # Initialize output matrix A
	    A = np.zeros((m, n_H, n_W, n_C))              
	    
	    ### START CODE HERE ###
	    for i in range(m):                         # loop over the training examples
	        for h in range(n_H):                     # loop on the vertical axis of the output volume
	            for w in range(n_W):                 # loop on the horizontal axis of the output volume
	                for c in range (n_C):            # loop over the channels of the output volume
	                    
	                    # Find the corners of the current "slice" (≈4 lines)
	                    vert_start = h  * stride
	                    vert_end = vert_start + f
	                    horiz_start = w * stride
	                    horiz_end = horiz_start + f
	                    
	                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
	
	                    a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
	                    
	                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
	                    if mode == "max":
	                        A[i, h, w, c] = np.max(a_prev_slice)
	                    elif mode == "average":
	                        A[i, h, w, c] = np.mean(a_prev_slice)
	    
	    ### END CODE HERE ###
	    
	    # Store the input and hparameters in "cache" for pool_backward()
	    cache = (A_prev, hparameters)
	    
	    # Making sure your output shape is correct
	    assert(A.shape == (m, n_H, n_W, n_C))
	    
	    return A, cache

    #test code
	np.random.seed(1)
	A_prev = np.random.randn(2, 4, 4, 3)
	hparameters = {"stride" : 2, "f": 3}
	
	A, cache = pool_forward(A_prev, hparameters)
	print("mode = max")
	print("A =", A)
	print()
	A, cache = pool_forward(A_prev, hparameters, mode = "average")
	print("mode = average")
	print("A =", A)

执行结果：

mode = max    
A = [[[[ 1.74481176  0.86540763  1.13376944]]]


 [[[ 1.13162939  1.51981682  2.18557541]]]]

mode = average    
A = [[[[ 0.02105773 -0.20328806 -0.40389855]]]


 [[[-0.22154621  0.51716526  0.48155844]]]]


Congratulations! You have now implemented the forward passes of all the layers of a convolutional network. 

The remainer of this notebook is optional, and will not be graded.