---
layout:     post
title:      深度卷积神经网络模型（测试题）
keywords:   博客
categories: [机器学习]
tags:	    [卷积神经网络，ConvNets]
---



## 第1题

Which of the following do you typically see as you move to deeper layers in a ConvNet?



- n\_H and n\_W increases, while n\_C decreases
- n\_H and n\_W decreases, while n\_C also decreases
- n\_H and n\_W decreases, while n\_C increases
- n\_H and n\_W increases, while n\_C also decreases

正确答案： 第3项。深度卷积网络，一般来说，网络越深，高度和宽度会减小，通道数会增加。

-------------------------------------

## 第2题

Which of the following do you typically see in a ConvNet? (Check all that apply.)

- Multiple CONV layers followed by a POOL layer
- Multiple POOL layers followed by a CONV layer
- FC layers in the last few layers
- FC layers in the first few layers   

正确答案：第1、3项。以AlexNet网络模型为例，多个卷积层conv后接着一个池化层Pool，最后是几个全连接层FC。其它的网络模型也基本遵循这种结构。

---------------------------------------------

## 第3题  

In order to be able to build very deep networks, we usually only use pooling layers to downsize the height/width of the activation volumes while convolutions are used with “valid” padding. Otherwise, we would downsize the input of the model too quickly.


- True
- False

正确答案：False。如果希望模型不会快速变小，应该使用"same" padding替代"valid" padding。

---------------------------------------

## 第4题 

Training a deeper network (for example, adding additional layers to the network) allows the network to fit more complex functions and thus almost always results in lower training error. For this question, assume we’re referring to “plain” networks.

- True
- False


正确答案： False。"plain"网络可以理解为正常的神经网络（非残差网络），理论上层数越多，训练误差会越小；实际上层数越多，意味着用优化算法训练越难，因此训练误差也会有所增加。

---------------------------------------

## 第5题

The following equation captures the computation in a ResNet block. What goes into the two blanks above?


   ![](/images/images_2018/8-20_01.png)   


正确答案： 第4个选项。

----------------------------------------------

## 第6题

Which ones of the following statements on Residual Networks are true? (Check all that apply.)

- A ResNet with L layers would have on the order of L2 skip connections in total.
- The skip-connections compute a complex non-linear function of the input to pass to a deeper layer in the network.
- Using a skip-connection helps the gradient to backpropagate and thus helps you to train deeper networks.
- The skip-connection makes it easy for the network to learn an identity mapping between the input and the output within the ResNet block.


正确答案： 第3、4个选项。这道题比较有迷惑性。

------------------------------------------

## 第7题

Suppose you have an input volume of dimension 64x64x16. How many parameters would a single 1x1 convolutional filter have (including the bias)?

- 1
- 2
- 4097
- 17

正确答案：第4个选项。16+1 = 17。

----------------------------------------

## 第8题

Suppose you have an input volume of dimension n_H * n_W * n_C. Which of the following statements you agree with? (Assume that “1x1 convolutional layer” below always uses a stride of 1 and no padding.)

- You can use a pooling layer to reduce n_H, n_W and n_C. 
- You can use a pooling layer to reduce n_H, n_W, but not n_C.
- You can use a 1x1 convolutional layer to reduce n_H, n_W and n_C.
- You can use a 1x1 convolutional layer to reduce n_C, but not n_H, n_W.


正确答案：第2、4个选项。池化层往往跟在卷积层后面，通过平均池化或者最大池化的方法，将之前卷积层得到的特征图做一个聚合统计。假设输入层是112*112*3,池化窗口大小为2*2，池化后输出56*56*3。因此池化层可以有效缩小矩阵的尺寸，既可以加快计算速度，也可以防止过拟合。1x1 卷积也称为网络中的网络，如果输入矩阵的channels 较多，可以采用1x1 卷积对其channels进行压缩，而不会改变输入矩阵的高和宽。

-----------------------------

## 第9题

Which ones of the following statements on Inception Networks are true? (Check all that apply.)

- A single inception block allows the network to use a combination of 1x1, 3x3, 5x5 convolutions and pooling.
- Inception networks incorporates a variety of network architectures (similar to dropout, which randomly chooses a network architecture on each step) and thus has a similar regularizing effect as dropout.
- Making an inception network deeper (by stacking more inception blocks together) should not hurt training set performance.
- Inception blocks usually use 1x1 convolutions to reduce the input data volume’s size before applying 3x3 and 5x5 convolutions.

正确答案：第1、3项。

--------------------------------------

## 第10题

Which of the following are common reasons for using open-source implementations of ConvNets (both the model and/or weights)? Check all that apply.

- The same techniques for winning computer vision competitions, such as using multiple crops at test time, are widely used in practical deployments (or production system deployments) of ConvNets.
- Parameters trained for one computer vision task are often useful as pretraining for other computer vision tasks.
- A model trained for one computer vision task can usually be used to perform data augmentation even for a different computer vision task.
- It is a convenient way to get working an implementation of a complex ConvNet architecture.

正确答案：第2、4个选项。



