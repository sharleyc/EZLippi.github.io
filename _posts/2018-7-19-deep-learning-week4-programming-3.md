---
layout:     post
title:      深层神经网络编程作业(三)  
keywords:   博客
categories: [机器学习]
tags:	    [深度学习，深层神经网络]
---

这个作业内容是在前面的基础上构建一个能识别猫的图片的深度神经网络。在之前的练习中，我们使用逻辑回归模型识别猫图片的准确率是68%，而这次的算法准确度能达到80%。这个作业可以帮助你：  

 - 学习如何利用辅助函数去构建一个你想要的模型。
 - 实验不同的模型，并观察它们的行为方式。
 - 在你从头开始构建神经网络之前，构建辅助函数能使你的工作更容易一些。  

![](/images/images_2018/backprop_kiank.png)  

## 1 - Packages

完成这个作业需要用到以下Python库，我们首先需要导入它们。
 
- [numpy](www.numpy.org) 
- [matplotlib](http://matplotlib.org) 
- [h5py](http://www.h5py.org) 
- [PIL](http://www.pythonware.com/products/pil/)
- [scipy](https://www.scipy.org/) 
- dnn\_app\_utils
- np.random.seed(1)

------------------------
	import time   
	import numpy as np   
	import h5py
	import matplotlib.pyplot as plt
	import scipy
	from PIL import Image
	from scipy import ndimage
	from dnn_app_utils_v3 import *
	
	%matplotlib inline
	plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
	plt.rcParams['image.interpolation'] = 'nearest'
	plt.rcParams['image.cmap'] = 'gray'
	
	%load_ext autoreload
	%autoreload 2
	
	np.random.seed(1)

---------------------------------

## 2 - Dataset

**Problem Statement**: 会给你一个数据集 ("data.h5") 它包含：  

   - 有标签的训练集图片m_train，1 表示cat，0 表示non-cat
   - 有标签的测试集图片m_test
   - 每张图片的形状是(num_px, num_px, 3)，其中3表示3个颜色通道 (RGB)

我们先了解一下数据集。以下代码的作用是加载数据。

    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()  

以下代码会显示数据集中的一张图片。你可以改变index的值，重新运行代码查看不同的图片。  

	# Example of a picture
	index = 10
	plt.imshow(train_x_orig[index])
	print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.") 

运行结果：  

   ![](/images/images_2018/7-19_01.jpg)    


	# Explore your dataset 
	m_train = train_x_orig.shape[0]
	num_px = train_x_orig.shape[1]
	m_test = test_x_orig.shape[0]
	
	print ("Number of training examples: " + str(m_train))
	print ("Number of testing examples: " + str(m_test))
	print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
	print ("train_x_orig shape: " + str(train_x_orig.shape))
	print ("train_y shape: " + str(train_y.shape))
	print ("test_x_orig shape: " + str(test_x_orig.shape))
	print ("test_y shape: " + str(test_y.shape))

运行结果： 

	Number of training examples: 209
	Number of testing examples: 50
	Each image is of size: (64, 64, 3)
	train_x_orig shape: (209, 64, 64, 3)
	train_y shape: (1, 209)
	test_x_orig shape: (50, 64, 64, 3)
	test_y shape: (1, 50)