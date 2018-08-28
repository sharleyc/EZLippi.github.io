---
layout:     post
title:      基于神经风格迁移的艺术生成(编程题)
keywords:   博客
categories: [机器学习]
tags:	    [卷积神经网络，神经风格迁移，艺术]
---

通过分离和重新组合图片内容与风格，卷积神经网络(CNN)可以创作出具有艺术魅力的作品。使用CNN将一张图片的语义内容与不同风格融合起来的过程被称为神经风格迁移(Neural Style Transfer)。


# Deep Learning & Art: Neural Style Transfer

Welcome to the second assignment of this week. In this assignment, you will learn about Neural Style Transfer. This algorithm was created by Gatys et al. (2015) (https://arxiv.org/abs/1508.06576). 

**In this assignment, you will:**

- Implement the neural style transfer algorithm 
- Generate novel artistic images using your algorithm 

Most of the algorithms you've studied optimize a cost function to get a set of parameter values. In Neural Style Transfer, you'll optimize a cost function to get pixel values!

这个作业你将完成神经风格迁移算法，并使用你的算法产生新颖的艺术图像。

你研究的大多数算法都是通过优化成本函数来获取一组参数值。在神经风格迁移中，你将优化成本函数来获取像素值。


	import os
	import sys
	import scipy.io
	import scipy.misc
	import matplotlib.pyplot as plt
	from matplotlib.pyplot import imshow
	from PIL import Image
	from nst_utils import *
	import numpy as np
	import tensorflow as tf
	
	%matplotlib inline


In this example, you are going to generate an image of the Louvre museum in Paris (content image C), mixed with a painting by Claude Monet, a leader of the impressionist movement (style image S).

   ![](/images/images_2018/softmax-regression-vectorequation.png)

Let's see how you can do this. 

在这个例子中，你将巴黎卢浮宫的图像(内容图像C)，与印象派领袖人物的一幅画作(风格图像S)融合生成一张新的图片。



## 2 - Transfer Learning

Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. The idea of using a network trained on a different task and applying it to a new task is called transfer learning. 

Following the original NST paper (https://arxiv.org/abs/1508.06576), we will use the VGG network. Specifically, we'll use VGG-19, a 19-layer version of the VGG network. This model has already been trained on the very large ImageNet database, and thus has learned to recognize a variety of low level features (at the earlier layers) and high level features (at the deeper layers). 

Run the following code to load parameters from the VGG model. This may take a few seconds. 


神经风格迁移(NST)在已经训练好的卷积网络上构建。将不同任务上训练得到的网络应用于新任务的做法称为迁移学习。根据NST论文，我们将使用VGG网络。具体来说是VGG-19，这是一个19层版本的VGG网络。该模型已经在非常大的ImageNet数据库上进行了训练，因此学会了识别各种低级特征(较早的层)和高级特征（更深的层）。

运行以下代码从VGG模型加载参数。

	model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
	print(model)

The model is stored in a python dictionary where each variable name is the key and the corresponding value is a tensor containing that variable's value. To run an image through this network, you just have to feed the image to the model. In TensorFlow, you can do so using the [tf.assign](https://www.tensorflow.org/api_docs/python/tf/assign) function. In particular, you will use the assign function like this:  
```
model["input"].assign(image)
```

This assigns the image as an input to the model. After this, if you want to access the activations of a particular layer, say layer `4_2` when the network is run on this image, you would run a TensorFlow session on the correct tensor `conv4_2`, as follows:  
```
sess.run(model["conv4_2"])
```

## 3 - Neural Style Transfer 

We will build the NST algorithm in three steps:

- Build the content cost function J<sub>content</sub>(C,G)
- Build the style cost function J<sub>style</sub>(S,G)
- Put it together to get J(G) = &alpha;J<sub>content</sub>(C,G) + &beta; J<sub>style</sub>(S,G). 

NST算法的三个步骤：

- 构建内容成本函数：J<sub>content</sub>(C,G)
- 构建风格成本函数：J<sub>style</sub>(S,G)
- 组合内容成本函数和风格成本函数：
J(G) = &alpha;J<sub>content</sub>(C,G) + &beta; J<sub>style</sub>(S,G)

### 3.1 - Computing the content cost

In our running example, the content image C will be the picture of the Louvre Museum in Paris. Run the code below to see a picture of the Louvre.   
运行以下代码可以看到内容图像：   

	content_image = scipy.misc.imread("images/louvre.jpg")
	imshow(content_image)

   ![](/images/images_2018/content_louvre.png)   

The content image (C) shows the Louvre museum's pyramid surrounded by old Paris buildings, against a sunny sky with a few clouds.


#### 3.1.1 - How do you ensure the generated image G matches the content of the image C?

As we saw in lecture, the earlier (shallower) layers of a ConvNet tend to detect lower-level features such as edges and simple textures, and the later (deeper) layers tend to detect higher-level features such as more complex textures as well as object classes. 

ConvNet的浅层倾向于检测较低级别的特征，比如边缘和简单纹理，而深层倾向于检测更高级的特征，比如更复杂的纹理对象。

We would like the "generated" image G to have similar content as the input image C. Suppose you have chosen some layer's activations to represent the content of an image. In practice, you'll get the most visually pleasing results if you choose a layer in the middle of the network--neither too shallow nor too deep. (After you have finished this exercise, feel free to come back and experiment with using different layers, to see how the results vary.)

我们希望生成的图像G具有输入图像C相似的内容。假设你已经选择了某个图层的激活来表示图像的内容，在实践中，如果你选择的层既不太浅也不太深，那么将获得最令人满意的结果。（完成练习后，请随时回来尝试使用不同的层，看看结果如何变化）

So, suppose you have picked one particular hidden layer to use. Now, set the image C as the input to the pretrained VGG network, and run forward propagation. Let a<sup>(C)</sup> be the hidden layer activations in the layer you had chosen. (In lecture, we had written this as a<sup>[l]</sup><sup>(C)</sup>, but here we'll drop the superscript [l] to simplify the notation.) This will be a n\_H &times; n\_W &times; n\_C tensor. Repeat this process with the image G: Set G as the input, and run forward progation. Let a<sup>(G)</sup> be the corresponding hidden layer activation. We will define as the content cost function as:


   ![](/images/images_2018/8-28_01.jpg)

Here, n\_H, n\_W and n\_C are the height, width and number of channels of the hidden layer you have chosen, and appear in a normalization term in the cost. For clarity, note that a<sup>(C)</sup> and $a^{(G)}$ are the volumes corresponding to a hidden layer's activations. In order to compute the cost $J_{content}(C,G)$, it might also be convenient to unroll these 3D volumes into a 2D matrix, as shown below. (Technically this unrolling step isn't needed to compute $J_{content}$, but it will be good practice for when you do need to carry out a similar operation later for computing the style const $J_{style}$.)


