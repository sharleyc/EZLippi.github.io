---
layout:     post
title:      面部识别和神经风格转换（测试题）
keywords:   博客
categories: [机器学习]
tags:	    [卷积神经网络，面部识别，神经风格转换]
---



## 第1题

Face verification requires comparing a new picture against one person’s face, whereas face recognition requires comparing a new picture against K person’s faces.

- True
- False


正确答案： True。人脸验证(Face verification)是判断两个人脸图是否为同一人的算法，是一对一的问题；而人脸识别(Face Recognition)是识别出输入人脸图对应身份的算法，它通过和注册在资料库中的K个身份对应的特征进行逐个比对，找出差异度最小的特征，如果差异度都很大，则返回不在资料库中。

-------------------------------------

## 第2题

Why do we learn a function d(img1, img2) for face verification? (Select all that apply.)

- We need to solve a one-shot learning problem.
- This allows us to learn to recognize a new person given just a single image of that person.
- This allows us to learn to predict a person’s identity using a softmax output unit, where the number of classes equals the number of persons in the database plus 1 (for the final “not in database” class).
- Given how few images we have per person, we need to apply transfer learning.   

正确答案：第1、2项。人脸识别的挑战之一是单样本(one-shot)学习问题。即你必须从一个训练样本中学习，即可以做到能认出这个人。你可以使用softmax的输出单元对应资料库中的每个人或不是任何一个，但是因为样本太少，这样做的效果并不好。因此你需要学习一个相似度函数，能够输出两个图像之间的差异程度，当差异程度小于一个阀值，你就会推断两张照片是同一个人。

---------------------------------------------

## 第3题  

In order to train the parameters of a face recognition system, it would be reasonable to use a training set comprising 100,000 pictures of 100,000 different


- True
- False

正确答案：False。为了获得一个优良的人脸图片的编码，有一种方法是定义一个应用了梯度下降的三元组损失函数(triplet loss function)。要使用三元组损失，你需要同时查看三张照片，分别是锚照片A(anchor)，正例照片P(positive)和负例照片N(negative)。正例照片与锚照片的人相同，而负例照片和锚照片的人不同。训练中你需要一些成对的A和P，即一对包含同一人的照片，为了达到训练目的，必须要同一个人有数张不同照片。如果每个人只有一张照片，就无法训练你的系统。

---------------------------------------

## 第4题 

Which of the following is a correct definition of the triplet loss? Consider that α>0. (We encourage you to figure out the answer from first principles, rather than just refer to the lecture.)

- max(∣∣f(A)−f(P)∣∣<sup>2</sup> −∣∣f(A)−f(N)∣∣<sup>2</sup> + α, 0)
- max(∣∣f(A)−f(P)∣∣<sup>2</sup> −∣∣f(A)−f(N)∣∣<sup>2</sup> - α, 0)
- max(∣∣f(A)−f(N)∣∣<sup>2</sup> −∣∣f(A)−f(P)∣∣<sup>2</sup> - α, 0)
- max(∣∣f(A)−f(N)∣∣<sup>2</sup> −∣∣f(A)−f(P)∣∣<sup>2</sup> + α, 0)


正确答案： 第一项。我们希望∣∣f(A)−f(P)∣∣<sup>2</sup> <= ∣∣f(A)−f(N)∣∣<sup>2</sup>，即∣∣f(A)−f(P)∣∣<sup>2</sup> - ∣∣f(A)−f(N)∣∣<sup>2</sup> <= 0，为了防止神经网络输出退化解，我们想要上述这个式子小于负alpha。我们通常在左边写成正alpha，而不是右边的负alpha，即∣∣f(A)−f(P)∣∣<sup>2</sup> - ∣∣f(A)−f(N)∣∣<sup>2</sup> + α <= 0。损失函数的定义即为，整个式子的结果和0之间的最大值。即只要损失小于0，那么损失便为0，神经网络并不需要考虑具体负多少。而当损失大于0时，通过尝试最小化损失，对应的结果一定会等于0或小于0。

---------------------------------------

## 第5题

Consider the following Siamese network architecture:


   ![](/images/images_2018/8-27_01.png)   

The upper and lower neural networks have different input images, but have exactly the same parameters.


- True
- False
 

正确答案： True。孪生网络(Siamese Network)，即用两个完全相同的卷积神经网络对两张不同的图片进行计算，比较两者的结果。既然是完全相同的网络，那么参数也是完全相同的。

----------------------------------------------

## 第6题

You train a ConvNet on a dataset with 100 different classes. You wonder if you can find a hidden unit which responds strongly to pictures of cats. (I.e., a neuron so that, of all the input/training images that strongly activate that neuron, the majority are cat pictures.) You are more likely to find this unit in layer 4 of the network than in layer 1.

- True
- False


正确答案： True。深度卷积神经网络中的第一层的隐藏单元，通常在寻找相对简单的特征，比如边或特别的颜色；在更深的层中，将看到更大一部分图像，能检测出来的特征或模式更为复杂。因此第4层比第1层更容易检测出猫这样复杂的东西。

------------------------------------------

## 第7题

Neural style transfer is trained as a supervised learning task in which the goal is to input two images (x), and train a network to output a new, synthesized image (y).

- True
- False

正确答案：False。为了构建一个神经风格转移系统，我们为生成的图像定义一个代价函数，通过最小化这个代价函数能够生成想要的图像。这个代价函数包含两个部分，一个部分是内容代价，它衡量生成图像的内容与内容图像的内容之间的相似程度；另一个部分是风格代价，它衡量生成图像的风格与风格图像的相似程度。

----------------------------------------

## 第8题

In the deeper layers of a ConvNet, each channel corresponds to a different feature detector. The style matrix G<sup>[l]</sup> measures the degree to which the activations of different feature detectors in layer l vary (or correlate) together with each other.

- True
- False

正确答案：True。我们将风格定义成第l层各个通道的启动值之间的相关度。假设红色的通道对应到检测垂直花样的神经元，黄色通道对应的神经元则是找出橘色系的图块。那么如果两个通道相关的话，表示无论在图片的哪个区域，还有垂直花样的时候，那个区域也很可能有橘色系的色调。相关度可以告诉你，这些抽象的花样材质在图片的某处是否会倾向于一起出现。这给了你一种衡量方法，看看生成图片和输入风格图片两者有多相似。风格成本函数G如果是在第l层，可以定义为S(Style image)和G(Generated image)两个矩阵的各个元素差距的平方和。

-----------------------------

## 第9题

In neural style transfer, what is updated in each iteration of the optimization algorithm?

- The pixel values of the content image C
- The regularization parameters
- The pixel values of the generated image G
- The neural network parameters

正确答案：第3项。在神经风格转移算法中，通过最小化代价函数来生成想要的图像。标准化常数不大重要。

--------------------------------------

## 第10题

You are working with 3D data. You are building a network layer whose input volume has size 32x32x32x16 (this volume has 16 channels), and applies convolutions with 32 filters of dimension 3x3x3 (no padding, stride 1). What is the resulting output volume?

- 30x30x30x32
- Undefined: This convolution step is impossible and cannot be performed because the dimensions specified don’t match up.
- 30x30x30x16


正确答案：第1个选项。三维的高度宽度深度的计算公式和二维的相似，32-3+1 = 30，而卷积操作的结果，两边的通道数要保持一致。因此选择第1个选项。



