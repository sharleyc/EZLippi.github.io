---
layout:     post
title:      机器学习基础之偏差和方差
keywords:   博客
categories: [机器学习]
tags:	    [Tensorflow, GPU, 安装]
---

在机器学习中，我们需要一种合理的诊断方法来判断模型是不是我们想要的。除了实验估计模型的泛化性能，偏差和方差提供了一种评估模型的通用方法。 


## 偏差和方差的直观感受

我们先从靶心图来对偏差和方差有一个直观的感受：

   ![](/images/images_2018/11-05_01.png)  

假设红色的靶心区域是算法完美的正确预测值，蓝色点为训练出来的模型对样本的预测值。蓝色点比较集中的属于方差小的情况，比较分散的属于方差大的情况。蓝色点区域靠近红色靶心的属于偏差小的情况，远离红色靶心的属于偏差大的情况。

## 偏差和方差的定义  

接下来，继续看一下偏差和方差的含义：  

偏差(Bias)度量了模型的输出预测结果的期望和真实值之间的偏离程度，刻画了算法本身的拟合能力。(模型的精准度)

方差(Variance)度量了同样大小的训练集的变动所导致的学习性能的变化，刻画了数据扰动造成的影响。(模型的稳定性)

我们希望偏差和方差越小越好。

## 欠拟合和过拟合

高偏差对应欠拟合，高方差对应过拟合。在训练过程中，我们可以根据训练集和开发集的数据来判断模型是欠拟合还是过拟合。

例一：训练集误差1%，开发集误差11%。这可能是模型过拟合了，模型对训练集处理得非常好，但开发集就不尽如人意，在某种程度上泛化性不好。我们将这种情况定义为高方差。

例二：训练集误差15%，开发集误差16%。模型并未对训练集数据处理得很好，就是欠拟合的。我们将这种情况定义为高偏差。  

例三：训练集误差15%，开发集误差30%。我们可以判断出这个算法是高偏差的，并且还是高方差的。

例四：训练集误差0.5%，开发集误差1%。这个算法是低偏差和低方差的，是理想的算法。

以上的前提是理想误差为0%的假设。综上，我们可以通过观察训练集的误差，知道算法是否有高偏差问题，可以通过观察开发集的误差，知道算法是否有高方差问题。


## 系统改进算法的基本准则

第一： 首先看模型是否有高偏差？即看模型在训练集上的表现。模型不能良好拟合训练集，说明模型有高偏差。可以尝试:

- 更大的网络，带有更多隐藏层或隐藏单元(总是有效)
- 延长训练时间
- 换用更高级的优化算法

第二： 当偏差减小到可以接受的范围后，再看模型是否有高方差？模型在训练集上性能良好，但在开发集上表现不尽如人意，说明模型有高方差。解决高方差的方法：       

 - 获得更多数据
 - 正则化
 - 更适合的神经网络结构
 
   



