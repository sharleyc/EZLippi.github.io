---
layout:     post
title:      Machine Learning基础系列（一）
keywords:   博客
categories: [Python]
tags:	    [ml]
---

2018年上半年自学了吴恩达在coursera上的machine learning网上课程。这个博客系列融合了该课程的读书笔记以及在python上的实践。我学习该课程的初衷是希望在机器学习上有个入门，对机器学习的基本概念和原理有所了解。从这个角度讲，该课程确实值得推荐，但如果你想学完这门课程就能直接做项目，还是远远不够的。

## 机器学习的定义    

课程中对机器学习给出了两个定义：     
第一个定义来自于Arthur Samuel（1959）。他定义机器学习为：在进行特定编程的情况下，给予计算机学习能力的领域   
（Field of study that gives computers the ability to learn without being explicitly programmed）    

另一个年代近一点的定义，是卡内基梅隆大学的Tom Mitchell提出的，其定义如下， 一个程序被认为能从经验E中学习，解决任务T，达到性能度量值P，当且仅当，有了经验E后，经过P评判，程序在处理T时的性能有所提升。   
（A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P,if its performance at tasks in T,as measured by P,improves with experience E）     

## 机器学习的分类    

广义上分为监督学习和非监督学习（Supervised learning and Unsupervised learning）    

术语监督学习，意指给出一个算法，需要部分数据集已经有正确答案。监督学习大致可划分为两类：回归和分类，前者是要预测连续的值，比如根据照片预测人的年龄，后者是预测离散的值，比如确定病人的肿瘤是良性的还是恶性的。     

无监督学习是一种学习机制，你给算法大量的数据，要求它找出数据中蕴含的类型结构。比如自动发现细分市场。无监督学习大致可分为两类：聚类和非聚类（鸡尾酒会算法）。

## 线性回归算法 
 
第一周的课程里介绍了第一种机器学习算法：线性回归算法。以预测房价为例，引入了线性回归算法。

  ![](/images/images_2018/week1_1.png)     
  
我们要寻找最适合训练集的假设h，h是一个x到y的函数映射。在线性回归算法中，h是一条直线。在这个模型中有两个参数值，选择不同的参数会得到不同的假设。我们要选择的参数值，要让h表示的直线尽量地和这些数据点很好的拟合，即让h(x)和y之间的差异最小化。这个函数就是代价函数(Cost function)：   

  ![](/images/images_2018/week1_2.png)   

代价函数也被称为平方误差函数，这所以选择这个函数，是因为对于大多数问题，特别是回归问题，平方误差代价函数是非常合理的选择，也是最常用的手段。