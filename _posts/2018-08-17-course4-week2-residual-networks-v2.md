---
layout:     post
title:      残差网络(编程)
keywords:   博客
categories: [机器学习]
tags:	    [Residual，ConvNets]
---

在之前的学习中，我们了解到CNN的网络层数越多，意味着能够提取到不同level的特征越丰富，网络越具有语义信息。如果简单地增加深度，会导致梯度消失或梯度爆炸。解决方法是正则化初始化和中间的正则化层(Batch Normalization)，但是这样又引入了退化问题，即随着网络层数增加，训练集上的准确率不再上升甚至下降了。怎么解决退化问题？深度残差网络。实验表明，残差网络更容易优化，并且能够通过增加相当的深度来提高准确度。其核心是解决了增加深度带来的副作用（退化问题），能够通过单纯地增加网络深度，来提高网络性能。

# Residual Networks

Welcome to the second assignment of this week! You will learn how to build very deep convolutional networks, using Residual Networks (ResNets). In theory, very deep networks can represent very complex functions; but in practice, they are hard to train. Residual Networks, introduced by [He et al.](https://arxiv.org/pdf/1512.03385.pdf), allow you to train much deeper networks than were previously practically feasible.

**In this assignment, you will:**
- Implement the basic building blocks of ResNets. 
- Put together these building blocks to implement and train a state-of-the-art neural network for image classification. 

This assignment will be done in Keras. 

Before jumping into the problem, let's run the cell below to load the required packages.

你将使用Keras完成本次作业，作业内容是实现并训练一个深度残差网络，达到对图片进行分类的目标。首先是引入所需的库文件。


	import numpy as np
	from keras import layers
	from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
	from keras.models import Model, load_model
	from keras.preprocessing import image
	from keras.utils import layer_utils
	from keras.utils.data_utils import get_file
	from keras.applications.imagenet_utils import preprocess_input
	import pydot
	from IPython.display import SVG
	from keras.utils.vis_utils import model_to_dot
	from keras.utils import plot_model
	from resnets_utils import *
	from keras.initializers import glorot_uniform
	import scipy.misc
	from matplotlib.pyplot import imshow
	%matplotlib inline
	
	import keras.backend as K
	K.set_image_data_format('channels_last')
	K.set_learning_phase(1)


## 1 - The problem of very deep neural networks

Last week, you built your first convolutional neural network. In recent years, neural networks have become deeper, with state-of-the-art networks going from just a few layers (e.g., AlexNet) to over a hundred layers.

The main benefit of a very deep network is that it can represent very complex functions. It can also learn features at many different levels of abstraction, from edges (at the lower layers) to very complex features (at the deeper layers). However, using a deeper network doesn't always help. A huge barrier to training them is vanishing gradients: very deep networks often have a gradient signal that goes to zero quickly, thus making gradient descent unbearably slow. More specifically, during gradient descent, as you backprop from the final layer back to the first layer, you are multiplying by the weight matrix on each step, and thus the gradient can decrease exponentially quickly to zero (or, in rare cases, grow exponentially quickly and "explode" to take very large values). 

During training, you might therefore see the magnitude (or norm) of the gradient for the earlier layers descrease to zero very rapidly as training proceeds: 


   ![](/images/images_2018/vanishing_grad_kiank.png) 

You are now going to solve this problem by building a Residual Network!

## 2 - Building a Residual Network

In ResNets, a "shortcut" or a "skip connection" allows the gradient to be directly backpropagated to earlier layers:  


   ![](/images/images_2018/skip_connection_kiank.png)


<center>**Figure 2** : A ResNet block showing a **skip-connection** <br> </center>

The image on the left shows the "main path" through the network. The image on the right adds a shortcut to the main path. By stacking these ResNet blocks on top of each other, you can form a very deep network. 

We also saw in lecture that having ResNet blocks with the shortcut also makes it very easy for one of the blocks to learn an identity function. This means that you can stack on additional ResNet blocks with little risk of harming training set performance. (There is also some evidence that the ease of learning an identity function--even more than skip connections helping with vanishing gradients--accounts for ResNets' remarkable performance.)

Two main types of blocks are used in a ResNet, depending mainly on whether the input/output dimensions are same or different. You are going to implement both of them. 


### 2.1 - The identity block

The identity block is the standard block used in ResNets, and corresponds to the case where the input activation (say $a^{[l]}$) has the same dimension as the output activation (say $a^{[l+2]}$). To flesh out the different steps of what happens in a ResNet's identity block, here is an alternative diagram showing the individual steps:

   ![](/images/images_2018/idblock2_kiank.png)

<center> **Figure 3** : **Identity block.** Skip connection "skips over" 2 layers. </center>

The upper path is the "shortcut path." The lower path is the "main path." In this diagram, we have also made explicit the CONV2D and ReLU steps in each layer. To speed up training we have also added a BatchNorm step. Don't worry about this being complicated to implement--you'll see that BatchNorm is just one line of code in Keras! 

In this exercise, you'll actually implement a slightly more powerful version of this identity block, in which the skip connection "skips over" 3 hidden layers rather than 2 layers. It looks like this: 


   ![](/images/images_2018/idblock3_kiank.png)


<center> **Figure 4** : **Identity block.** Skip connection "skips over" 3 layers. </center>

Here're the individual steps.

First component of main path:     

- The first CONV2D has $F_1$ filters of shape (1,1) and a stride of (1,1). Its padding is "valid" and its name should be `conv_name_base + '2a'`. Use 0 as the seed for the random initialization. 
- The first BatchNorm is normalizing the channels axis.  Its name should be `bn_name_base + '2a'`.
- Then apply the ReLU activation function. This has no name and no hyperparameters. 

Second component of main path:

- The second CONV2D has $F_2$ filters of shape $(f,f)$ and a stride of (1,1). Its padding is "same" and its name should be `conv_name_base + '2b'`. Use 0 as the seed for the random initialization. 
- The second BatchNorm is normalizing the channels axis.  Its name should be `bn_name_base + '2b'`.
- Then apply the ReLU activation function. This has no name and no hyperparameters. 

Third component of main path:

- The third CONV2D has $F_3$ filters of shape (1,1) and a stride of (1,1). Its padding is "valid" and its name should be `conv_name_base + '2c'`. Use 0 as the seed for the random initialization. 
- The third BatchNorm is normalizing the channels axis.  Its name should be `bn_name_base + '2c'`. Note that there is no ReLU activation function in this component. 

Final step: 

- The shortcut and the input are added together.
- Then apply the ReLU activation function. This has no name and no hyperparameters. 

**Exercise**: Implement the ResNet identity block. We have implemented the first component of the main path. Please read over this carefully to make sure you understand what it is doing. You should implement the rest. 

- To implement the Conv2D step: [See reference](https://keras.io/layers/convolutional/#conv2d)
- To implement BatchNorm: [See reference](https://faroit.github.io/keras-docs/1.2.2/layers/normalization/) (axis: Integer, the axis that should be normalized (typically the channels axis))
- For the activation, use:  `Activation('relu')(X)`
- To add the value passed forward by the shortcut: [See reference](https://keras.io/layers/merge/#add)


-------------------------------------------

	# GRADED FUNCTION: identity_block
	
	def identity_block(X, f, filters, stage, block):
	    """
	    Implementation of the identity block as defined in Figure 3
	    
	    Arguments:
	    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
	    f -- integer, specifying the shape of the middle CONV's window for the main path
	    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
	    stage -- integer, used to name the layers, depending on their position in the network
	    block -- string/character, used to name the layers, depending on their position in the network
	    
	    Returns:
	    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
	    """
	    
	    # defining name basis
	    conv_name_base = 'res' + str(stage) + block + '_branch'
	    bn_name_base = 'bn' + str(stage) + block + '_branch'
	    
	    # Retrieve Filters
	    F1, F2, F3 = filters
	    
	    # Save the input value. You'll need this later to add back to the main path. 
	    X_shortcut = X
	    
	    # First component of main path
	    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
	    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
	    X = Activation('relu')(X)
	    
	    ### START CODE HERE ###
	    
	    # Second component of main path (≈3 lines)
	    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
	    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
	    X = Activation('relu')(X)
	
	    # Third component of main path (≈2 lines)
	    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
	    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
	
	    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
	    X = Add()([X, X_shortcut])
	    X = Activation('relu')(X)
	    
	    ### END CODE HERE ###
	    
	    return X

    #test code
	tf.reset_default_graph()
	
	with tf.Session() as test:
	    np.random.seed(1)
	    A_prev = tf.placeholder("float", [3, 4, 4, 6])
	    X = np.random.randn(3, 4, 4, 6)
	    A = identity_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
	    test.run(tf.global_variables_initializer())
	    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
	    print("out = " + str(out[0][1][1][0]))

执行结果：

out = [ 0.94822985  0.          1.16101444  2.747859    0.          1.36677003]

#### 2.1.1 - keras.layers.Conv2D

filters: 卷积和的数目，即输出的维度。   
kernel_size：卷积核的宽度和长度。    
strides：卷积的步长。   
padding：'valid'表示有效卷积，即不处理边界数据；'same'表示输出shape和输入shape相同。   
kernel_initializer：用于初始化权重的初始化器。glorot_uniform是Glorot Initialization的均匀分布版本，它是Keras中全连接层、二维卷积/反卷积层等层的默认初始化方式。


#### 2.1.2 - keras.layers.BatchNormalization

在神经网络训练开始前，都要对输入数据做一个归一化处理，原因在于神经网络学习过程本质是为了学习数据分布，一旦训练数据与测试数据的分布不同，网络的泛化能力也大大降低，另一方面，每批训练数据的分布各不相同，网络就需要在每次迭代都去学习适应不同的分布，会大大降低网络的训练速度。Batch Normalization算法就是为了解决在训练过程中，中间层数据分布发生变化的情况。

#### 2.1.3 - 'Tensor' object has no attribute '\_keras\_history'  

这里有个坑，需要专门说明一下：  

	X = Add()([X, X_shortcut])

这行代码之前的写法是：

    X = X + X_shortcut  

这会导致2.2 convolution block的代码运行报错（即使代码内容是正确的）。  

参考https://github.com/keras-team/keras/issues/7362，通常以下两种情况下会出现这种错误提示： 

- your inputs don't come from `keras.layers.Input()`
- your outputs aren't the output of a Keras layer


## 2.2 - The convolutional block

You've implemented the ResNet identity block. Next, the ResNet "convolutional block" is the other type of block. You can use this type of block when the input and output dimensions don't match up. The difference with the identity block is that there is a CONV2D layer in the shortcut path: 


   ![](/images/images_2018/convblock_kiank.png)


<center>  **Figure 4** : **Convolutional block. ** </center>


The CONV2D layer in the shortcut path is used to resize the input $x$ to a different dimension, so that the dimensions match up in the final addition needed to add the shortcut value back to the main path. (This plays a similar role as the matrix $W_s$ discussed in lecture.) For example, to reduce the activation dimensions's height and width by a factor of 2, you can use a 1x1 convolution with a stride of 2. The CONV2D layer on the shortcut path does not use any non-linear activation function. Its main role is to just apply a (learned) linear function that reduces the dimension of the input, so that the dimensions match up for the later addition step. 

The details of the convolutional block are as follows. 

First component of main path:

- The first CONV2D has $F_1$ filters of shape (1,1) and a stride of (s,s). Its padding is "valid" and its name should be `conv_name_base + '2a'`. 
- The first BatchNorm is normalizing the channels axis.  Its name should be `bn_name_base + '2a'`.
- Then apply the ReLU activation function. This has no name and no hyperparameters. 

Second component of main path:

- The second CONV2D has $F_2$ filters of (f,f) and a stride of (1,1). Its padding is "same" and it's name should be `conv_name_base + '2b'`.
- The second BatchNorm is normalizing the channels axis.  Its name should be `bn_name_base + '2b'`.
- Then apply the ReLU activation function. This has no name and no hyperparameters. 

Third component of main path:

- The third CONV2D has $F_3$ filters of (1,1) and a stride of (1,1). Its padding is "valid" and it's name should be `conv_name_base + '2c'`.
- The third BatchNorm is normalizing the channels axis.  Its name should be `bn_name_base + '2c'`. Note that there is no ReLU activation function in this component. 

Shortcut path:

- The CONV2D has $F_3$ filters of shape (1,1) and a stride of (s,s). Its padding is "valid" and its name should be `conv_name_base + '1'`.
- The BatchNorm is normalizing the channels axis.  Its name should be `bn_name_base + '1'`. 

Final step: 

- The shortcut and the main path values are added together.
- Then apply the ReLU activation function. This has no name and no hyperparameters. 
    
**Exercise**: Implement the convolutional block. We have implemented the first component of the main path; you should implement the rest. As before, always use 0 as the seed for the random initialization, to ensure consistency with our grader.
- [Conv Hint](https://keras.io/layers/convolutional/#conv2d)
- [BatchNorm Hint](https://keras.io/layers/normalization/#batchnormalization) (axis: Integer, the axis that should be normalized (typically the features axis))
- For the activation, use:  `Activation('relu')(X)`
- [Addition Hint](https://keras.io/layers/merge/#add)


