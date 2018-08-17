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

这会导致3 Building your first ResNet model的代码运行报错（即使代码内容是正确的）。  

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


	# GRADED FUNCTION: convolutional_block
	
	def convolutional_block(X, f, filters, stage, block, s = 2):
	    """
	    Implementation of the convolutional block as defined in Figure 4
	    
	    Arguments:
	    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
	    f -- integer, specifying the shape of the middle CONV's window for the main path
	    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
	    stage -- integer, used to name the layers, depending on their position in the network
	    block -- string/character, used to name the layers, depending on their position in the network
	    s -- Integer, specifying the stride to be used
	    
	    Returns:
	    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
	    """
	    
	    # defining name basis
	    conv_name_base = 'res' + str(stage) + block + '_branch'
	    bn_name_base = 'bn' + str(stage) + block + '_branch'
	    
	    # Retrieve Filters
	    F1, F2, F3 = filters
	    
	    # Save the input value
	    X_shortcut = X
	
	
	    ##### MAIN PATH #####
	    # First component of main path 
	    X = Conv2D(F1, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
	    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
	    X = Activation('relu')(X)
	    
	    ### START CODE HERE ###
	
	    # Second component of main path (≈3 lines)
	    X = Conv2D(F2, (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
	    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
	    X = Activation('relu')(X)
	
	    # Third component of main path (≈2 lines)
	    X = Conv2D(F3, (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
	    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
	
	    ##### SHORTCUT PATH #### (≈2 lines)
	    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
	    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
	
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
	    A = convolutional_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
	    test.run(tf.global_variables_initializer())
	    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
	    print("out = " + str(out[0][1][1][0]))

执行结果：

out = [ 0.09018463  1.23489773  0.46822017  0.0367176   0.          0.65516603]


## 3 - Building your first ResNet model (50 layers)

You now have the necessary blocks to build a very deep ResNet. The following figure describes in detail the architecture of this neural network. "ID BLOCK" in the diagram stands for "Identity block," and "ID BLOCK x3" means you should stack 3 identity blocks together.

   ![](/images/images_2018/resnet_kiank.png)

<center> **Figure 5** : **ResNet-50 model** </center>

The details of this ResNet-50 model are:

- Zero-padding pads the input with a pad of (3,3)
- Stage 1:
    - The 2D Convolution has 64 filters of shape (7,7) and uses a stride of (2,2). Its name is "conv1".
    - BatchNorm is applied to the channels axis of the input.
    - MaxPooling uses a (3,3) window and a (2,2) stride.
- Stage 2:
    - The convolutional block uses three set of filters of size [64,64,256], "f" is 3, "s" is 1 and the block is "a".
    - The 2 identity blocks use three set of filters of size [64,64,256], "f" is 3 and the blocks are "b" and "c".
- Stage 3:
    - The convolutional block uses three set of filters of size [128,128,512], "f" is 3, "s" is 2 and the block is "a".
    - The 3 identity blocks use three set of filters of size [128,128,512], "f" is 3 and the blocks are "b", "c" and "d".
- Stage 4:
    - The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".
    - The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".
- Stage 5:
    - The convolutional block uses three set of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and the block is "a".
    - The 2 identity blocks use three set of filters of size [512, 512, 2048], "f" is 3 and the blocks are "b" and "c".
- The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".
- The flatten doesn't have any hyperparameters or name.
- The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation. Its name should be `'fc' + str(classes)`.

**Exercise**: Implement the ResNet with 50 layers described in the figure above. We have implemented Stages 1 and 2. Please implement the rest. (The syntax for implementing Stages 3-5 should be quite similar to that of Stage 2.) Make sure you follow the naming convention in the text above. 

You'll need to use this function: 
- Average pooling [see reference](https://keras.io/layers/pooling/#averagepooling2d)

Here're some other functions we used in the code below:
- Conv2D: [See reference](https://keras.io/layers/convolutional/#conv2d)
- BatchNorm: [See reference](https://keras.io/layers/normalization/#batchnormalization) (axis: Integer, the axis that should be normalized (typically the features axis))
- Zero padding: [See reference](https://keras.io/layers/convolutional/#zeropadding2d)
- Max pooling: [See reference](https://keras.io/layers/pooling/#maxpooling2d)
- Fully conected layer: [See reference](https://keras.io/layers/core/#dense)
- Addition: [See reference](https://keras.io/layers/merge/#add)


	# GRADED FUNCTION: ResNet50
	
	def ResNet50(input_shape = (64, 64, 3), classes = 6):
	    """
	    Implementation of the popular ResNet50 the following architecture:
	    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
	    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
	
	    Arguments:
	    input_shape -- shape of the images of the dataset
	    classes -- integer, number of classes
	
	    Returns:
	    model -- a Model() instance in Keras
	    """
	    
	    # Define the input as a tensor with shape input_shape
	    X_input = Input(input_shape)
	
	    
	    # Zero-Padding
	    X = ZeroPadding2D((3, 3))(X_input)
	    
	    # Stage 1
	    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
	    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
	    X = Activation('relu')(X)
	    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
	
	    # Stage 2
	    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
	    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
	    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
	
	    ### START CODE HERE ###
	
	    # Stage 3 (≈4 lines)
	    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
	    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
	    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
	    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
	
	    # Stage 4 (≈6 lines)
	    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
	    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
	    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
	    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
	    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
	    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
	
	    # Stage 5 (≈3 lines)
	    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
	    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
	    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
	
	    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
	    X = AveragePooling2D((2, 2),name='avg_pool')(X)
	    
	    ### END CODE HERE ###
	
	    # output layer
	    X = Flatten()(X)
	    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
	    
	    
	    # Create model
	    model = Model(inputs = X_input, outputs = X, name='ResNet50')
	
	    return model


Run the following code to build the model's graph. If your implementation is not correct you will know it by checking your accuracy when running model.fit(...) below.

    model = ResNet50(input_shape = (64, 64, 3), classes = 6)

As seen in the Keras Tutorial Notebook, prior training a model, you need to configure the learning process by compiling the model.

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

The model is now ready to be trained. The only thing you need is a dataset.

Let's load the SIGNS Dataset.


   ![](/images/images_2018/signs_data_kiank.png)


<center> **Figure 6**  : **SIGNS dataset** </center>


	X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
	
	# Normalize image vectors
	X_train = X_train_orig/255.
	X_test = X_test_orig/255.
	
	# Convert training and test labels to one hot matrices
	Y_train = convert_to_one_hot(Y_train_orig, 6).T
	Y_test = convert_to_one_hot(Y_test_orig, 6).T
	
	print ("number of training examples = " + str(X_train.shape[0]))
	print ("number of test examples = " + str(X_test.shape[0]))
	print ("X_train shape: " + str(X_train.shape))
	print ("Y_train shape: " + str(Y_train.shape))
	print ("X_test shape: " + str(X_test.shape))
	print ("Y_test shape: " + str(Y_test.shape))

执行结果：

number of training examples = 1080   
number of test examples = 120    
X_train shape: (1080, 64, 64, 3)    
Y_train shape: (1080, 6)   
X_test shape: (120, 64, 64, 3)   
Y_test shape: (120, 6)    


Run the following cell to train your model on 2 epochs with a batch size of 32. On a CPU it should take you around 5min per epoch. 

    model.fit(X_train, Y_train, epochs = 2, batch_size = 32)

执行结果：

Epoch 1/2   
1080/1080 [==============================] - 236s - loss: 3.1990 - acc: 0.2722       
Epoch 2/2   
1080/1080 [==============================] - 232s - loss: 2.3164 - acc: 0.3417     


**Expected Output**:

<table>
    <tr>
        <td>
            ** Epoch 1/2**
        </td>
        <td>
           loss: between 1 and 5, acc: between 0.2 and 0.5, although your results can be different from ours.
        </td>
    </tr>
    <tr>
        <td>
            ** Epoch 2/2**
        </td>
        <td>
           loss: between 1 and 5, acc: between 0.2 and 0.5, you should see your loss decreasing and the accuracy increasing.
        </td>
    </tr>

</table>


Let's see how this model (trained on only two epochs) performs on the test set.

	preds = model.evaluate(X_test, Y_test)
	print ("Loss = " + str(preds[0]))
	print ("Test Accuracy = " + str(preds[1]))

执行结果：

120/120 [==============================] - 8s         
Loss = 2.49306329091    
Test Accuracy = 0.166666666667   

我们可以看出测试结果很差，因为仅训练了两个迭代。如果要获得更好的结果，需要增加迭代次数，比如20次，如果是在CPU上训练接近一个小时，但是如果是在GPU上训练大约只要1分钟。


For the purpose of this assignment, we've asked you to train the model only for two epochs. You can see that it achieves poor performances. Please go ahead and submit your assignment; to check correctness, the online grader will run your code only for a small number of epochs as well.

After you have finished this official (graded) part of this assignment, you can also optionally train the ResNet for more iterations, if you want. We get a lot better performance when we train for ~20 epochs, but this will take more than an hour when training on a CPU. 

Using a GPU, we've trained our own ResNet50 model's weights on the SIGNS dataset. You can load and run our trained model on the test set in the cells below. It may take ≈1min to load the model.


    model = load_model('ResNet50.h5') 

	preds = model.evaluate(X_test, Y_test)
	print ("Loss = " + str(preds[0]))
	print ("Test Accuracy = " + str(preds[1]))


ResNet50 is a powerful model for image classification when it is trained for an adequate number of iterations. We hope you can use what you've learnt and apply it to your own classification problem to perform state-of-the-art accuracy.

Congratulations on finishing this assignment! You've now implemented a state-of-the-art image classification system!



<font color='blue'>
**What you should remember:**

- Very deep "plain" networks don't work in practice because they are hard to train due to vanishing gradients.  
- The skip-connections help to address the Vanishing Gradient problem. They also make it easy for a ResNet block to learn an identity function. 
- There are two main type of blocks: The identity block and the convolutional block. 
- Very deep Residual Networks are built by stacking these blocks together.


