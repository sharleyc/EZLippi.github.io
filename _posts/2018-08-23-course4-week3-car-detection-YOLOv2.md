---
layout:     post
title:      用YOLOv2检测汽车（编程作业）
keywords:   博客
categories: [机器学习]
tags:	    [卷积神经网络，自动驾驶]
---


通过本次编程作业，你将学习如何使用YOLO模型来进行物体检测。这里很多思想来自YOLO的两篇论文： Redmon et al., 2016 (https://arxiv.org/abs/1506.02640) and Redmon and Farhadi, 2016 (https://arxiv.org/abs/1612.08242)。

**你将学会**:   

- 在一个汽车检测数据集上进行物体检测(object detection)   
- 处理边界框(bounding boxes)

首先运行以下代码加载所需的库文件。

--------------------------------------

	import argparse
	import os
	import matplotlib.pyplot as plt
	from matplotlib.pyplot import imshow
	import scipy.io
	import scipy.misc
	import numpy as np
	import pandas as pd
	import PIL
	import tensorflow as tf
	from keras import backend as K
	from keras.layers import Input, Lambda, Conv2D
	from keras.models import load_model, Model
	from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
	from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
	
	%matplotlib inline


Important Note: As you can see, we import Keras's backend as K. This means that to use a Keras function in this notebook, you will need to write: K.function(...).


## 1 - Problem Statement

You are working on a self-driving car. As a critical component of this project, you'd like to first build a car detection system. To collect data, you've mounted a camera to the hood (meaning the front) of the car, which takes pictures of the road ahead every few seconds while you drive around. 

You've gathered all these images into a folder and have labelled them by drawing bounding boxes around every car you found. Here's an example of what your bounding boxes look like.

   ![](/images/images_2018/box_label.png)


If you have 80 classes that you want YOLO to recognize, you can represent the class label $c$ either as an integer from 1 to 80, or as an 80-dimensional vector (with 80 numbers) one component of which is 1 and the rest of which are 0. The video lectures had used the latter representation; in this notebook, we will use both representations, depending on which is more convenient for a particular step.  

In this exercise, you will learn how YOLO works, then apply it to car detection. Because the YOLO model is very computationally expensive to train, we will load pre-trained weights for you to use.



问题陈述： 自动驾驶项目中非常关键的一个部分是汽车检测系统，汽车前方的摄像头每隔几秒会拍摄前方图像，你收集了这些图片数据，并标注了图片中汽车的边界框。通过本次练习，你将学习YOLO如何工作，并将它应用到汽车检测中。

## 2 - YOLO

YOLO ("you only look once") is a popular algoritm because it achieves high accuracy while also being able to run in real-time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

YOLO是一个非常流行的算法，因为它不仅有较高的准确率，而且能实时运行。算法仅需要执行一次前向传播，在抑制非最大值算法执行结束后，就能输出带有边界框的检测物体。


### 2.1 - Model details

First things to know:

- The **input** is a batch of images of shape (m, 608, 608, 3)
- The **output** is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers $(p_c, b_x, b_y, b_h, b_w, c)$ as explained above. If you expand $c$ into an 80-dimensional vector, each bounding box is then represented by 85 numbers. 

We will use 5 anchor boxes. So you can think of the YOLO architecture as the following: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).

Lets look in greater detail at what this encoding represents. 

   ![](/images/images_2018/architecture.png)


