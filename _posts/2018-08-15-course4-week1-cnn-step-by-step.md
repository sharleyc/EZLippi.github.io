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

