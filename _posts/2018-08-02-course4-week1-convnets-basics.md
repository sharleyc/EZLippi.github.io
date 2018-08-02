---
layout:     post
title:      卷积神经网络-基础（测试题）
keywords:   博客
categories: [机器学习]
tags:	    [卷积神经网络，ConvNets]
---


## 第1题

What do you think applying this filter to a grayscale image will do?

   ![](/images/images_2018/8-2_01.png) 

- Detect image contrast
- Detect 45 degree edges
- Detect vertical edges
- Detect horizontal edges

正确答案： 第3项。这题有一定的迷惑性，仔细观察矩阵发现，其基本列对称，符号相反，因此是用来实现垂直边缘检测的。

-------------------------------------

## 第2题

Suppose your input is a 300 by 300 color (RGB) image, and you are not using a convolutional network. If the first hidden layer has 100 neurons, each one fully connected to the input, how many parameters does this hidden layer have (including the bias parameters)?

- 9,000,001
- 9,000,100
- 27,000,001
- 27,000,100   

正确答案：第4项。300 * 300 * 3 * 100 + 100 = 27,000,100。

---------------------------------------------

## 第3题  

Suppose your input is a 300 by 300 color (RGB) image, and you use a convolutional layer with 100 filters that are each 5x5. How many parameters does this hidden layer have (including the bias parameters)?


- 2501
- 2600
- 7500
- 7600

正确答案：第4个选项。一个过滤器的参数是5*5*3=75，加上bias就是76，100个过滤器就是76*100，因此答案是7600。

---------------------------------------

## 第4题 

You have an input volume that is 63x63x16, and convolve it with 32 filters that are each 7x7, using a stride of 2 and no padding. What is the output volume?

- 16x16x32
- 16x16x16
- 29x29x32
- 29x29x16

正确答案： 第3个选项。（63-7）/2 + 1 = 29，卷积计算输出的通道数等于过滤器的个数，因此是29x29x32。

---------------------------------------

## 第5题

You have an input volume that is 15x15x8, and pad it using “pad=2.” What is the dimension of the resulting volume (after padding)?

- 17x17x10
- 19x19x8
- 17x17x8
- 19x19x12     


正确答案： 第2个选项。填充后宽高都增加2P，通道数量不受影响。

----------------------------------------------

## 第6题

You have an input volume that is 63x63x16, and convolve it with 32 filters that are each 7x7, and stride of 1. You want to use a “same” convolution. What is the padding?

- 1
- 2
- 3
- 7


正确答案： 第3个选项。p = (f - 1)/2 = (7 - 1)/2 = 3。

------------------------------------------

## 第7题

You have an input volume that is 32x32x16, and apply max pooling with a stride of 2 and a filter size of 2. What is the output volume?

- 16x16x8
- 16x16x16
- 15x15x16
- 32x32x8

正确答案：第2项。（32 - 2）/2 + 1 = 16，不改变通道数。

----------------------------------------

## 第8题

Because pooling layers do not have parameters, they do not affect the backpropagation (derivatives) calculation.

- True
- False  


正确答案：False。池化层没有需要求解的参数，但是有超参数，比如过滤器大小、步长等。同样影响反向传播梯度运算的结果。

-----------------------------

## 第9题

In lecture we talked about “parameter sharing” as a benefit of using convolutional networks. Which of the following statements about parameter sharing in ConvNets are true? (Check all that apply.)

- It allows a feature detector to be used in multiple locations throughout the whole input image/input volume.
- It allows gradient descent to set many of the parameters to zero, thus making the connections sparse.
- It reduces the total number of parameters, thus reducing overfitting.
- It allows parameters learned for one task to be shared even for a different task (transfer learning).

正确答案：第1、3项。

--------------------------------------

## 第10题

In lecture we talked about “sparsity of connections” as a benefit of using convolutional layers. What does this mean?

- Each activation in the next layer depends on only a small number of activations from the previous layer.
- Each layer in a convolutional network is connected only to two other layers
- Each filter is connected to every channel in the previous layer.
- Regularization causes gradient descent to set many of the parameters to zero.

正确答案：第1项。

----------------------------------------

