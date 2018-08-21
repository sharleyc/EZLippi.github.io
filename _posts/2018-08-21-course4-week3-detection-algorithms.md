---
layout:     post
title:      深度卷积神经网络模型（测试题）
keywords:   博客
categories: [机器学习]
tags:	    [卷积神经网络，ConvNets]
---



## 第1题

You are building a 3-class object classification and localization algorithm. The classes are: pedestrian (c=1), car (c=2), motorcycle (c=3). What would be the label for the following image? Recall y = [p<sub>c</sub>, b<sub>x</sub>, b<sub>y</sub>, b<sub>h</sub>, b<sub>w</sub>, c<sub>1</sub>, c<sub>2</sub>, c<sub>3</sub>]   

   ![](/images/images_2018/8-21_01.png)

- y=[1,0.3,0.7,0.3,0.3,0,1,0]
- y=[1,0.7,0.5,0.3,0.3,0,1,0]
- y=[1,0.3,0.7,0.5,0.5,0,1,0]
- y=[1,0.3,0.7,0.5,0.5,1,0,0]
- y=[0,0.2,0.4,0.5,0.5,0,1,0]

正确答案： 第1项。解题方法可以采取排除法。这是一个目标分类和定位的问题。三个分类：行人，汽车，摩托车。显然，图片中是一辆汽车，故向量y=[1,？,？,？,？,0,1,0]；然后，汽车在图片中的位置是偏左下角的，因此y=[1,0.3,0.7,？,？,0,1,0]；最后，汽车的高度和宽度都不足图片的一半，因此选项 y=[1,0.3,0.7,0.3,0.3,0,1,0] 是最合适的。

-------------------------------------

## 第2题

Continuing from the previous problem, what should y be for the image below? Remember that “?” means “don’t care”, which means that the neural network loss function won’t care what the neural network gives for that component of the output. As before, y = [p<sub>c</sub>, b<sub>x</sub>, b<sub>y</sub>, b<sub>h</sub>, b<sub>w</sub>, c<sub>1</sub>, c<sub>2</sub>, c<sub>3</sub>]   

   ![](/images/images_2018/8-21_02.png)

- y=[1,?,?,?,?,0,0,0]
- y=[0,?,?,?,?,?,?,?]
- y=[1,?,?,?,?,?,?,?]
- y=[0,?,?,?,?,0,0,0]
- y=[?,?,?,?,?,?,?,?]

正确答案：第2项。这题接着上一题，图片里没有需要检测的目标(行人，汽车和摩托车)，因此除了p<sub>c</sub> = 0外，其它值都"dont't care"，即为？。因此选第2项。

---------------------------------------------

## 第3题  

You are working on a factory automation task. Your system will see a can of soft-drink coming down a conveyor belt, and you want it to take a picture and decide whether (i) there is a soft-drink can in the image, and if so (ii) its bounding box. Since the soft-drink can is round, the bounding box is always square, and the soft drink can always appears as the same size in the image. There is at most one soft drink can in each image. Here’re some typical images in your training set:

   ![](/images/images_2018/8-21_03.png)

What is the most appropriate set of output units for your neural network?


- Logistic unit (for classifying if there is a soft-drink can in the image)	
- Logistic unit,  b<sub>x</sub> and b<sub>y</sub>
- Logistic unit,  b<sub>x</sub> , b<sub>y</sub>, b<sub>h</sub>(since b<sub>h</sub> = b<sub>w</sub>)
- Logistic unit,  b<sub>x</sub> , b<sub>y</sub>, b<sub>h</sub>, b<sub>w</sub>

正确答案：第2项。目标物件的大小和形状是确定的，即b<sub>h</sub>, b<sub>w</sub>是已知的，不需要作为神经网络的输出单元。有无目标物件，以及物件的位置是未知的，因此选第2项。

---------------------------------------

## 第4题 

If you build a neural network that inputs a picture of a person’s face and outputs N landmarks on the face (assume the input image always contains exactly one face), how many output units will the network have?

- N
- 2N
- 3N
- N<sup>2</sup>


正确答案： 第2项。1个 "landmark" 需要b<sub>x</sub> , b<sub>y</sub>两个输出单元，因此N个 "landmark" 需要2N 个输出单元。

---------------------------------------

## 第5题

When training one of the object detection systems described in lecture, you need a training set that contains many pictures of the object(s) you wish to detect. However, bounding boxes do not need to be provided in the training set, since the algorithm can learn to detect the objects by itself.


- True
- False


正确答案： False。训练集的边界框(bounding boxes)需要提供来评估检测算法的误差。

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



