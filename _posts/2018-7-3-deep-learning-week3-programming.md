---
layout:     post
title:      深度学习之浅层神经网络--编程作业
keywords:   博客
categories: [机器学习]
tags:	    [深度学习，浅层神经网络]
---

编程作业的主题是具有一个隐藏层的平面数据分类。  

  ![](/images/images_2018/7-3_01.png) 


## 库文件       

导入完成这个任务需要的库文件：     

  ![](/images/images_2018/7-3_02.png)     

  ![](/images/images_2018/7-3_03.png)         


## 数据集    

  ![](/images/images_2018/7-3_04.png)   

  ![](/images/images_2018/7-3_05.png) 

  ![](/images/images_2018/7-3_06.png)   

### 通过np.shape函数获取数据集的属性     

  ![](/images/images_2018/7-3_07.png)   
    

## 简单的逻辑回归模型  

我们先通过逻辑回归模型来解决这个问题，通过sklearn的内置函数来训练逻辑回归分类器。      

  ![](/images/images_2018/7-3_08.png)    

最终的逻辑回归模型的准确率只有47% 。

  ![](/images/images_2018/7-3_09.png) 
 
从图中，可以明显得看出这个数据集是非线性分布的，因此逻辑回归模型的表现并不好。下面我们将尝试用神经网络来解决这个问题。         


## 神经网络模型    

首先回归一下神经网络的基本概念和公式： 

  ![](/images/images_2018/7-3_10.png)     

  ![](/images/images_2018/7-3_11.png)   

### 定义神经网络的结构  

这里定义了三个变量n_x,n_h,n_y，分别对应输入层，隐藏层和输出层。 

  ![](/images/images_2018/7-3_12.png)    
  
  ![](/images/images_2018/7-3_13.png)     

### 初始化模型参数  

  ![](/images/images_2018/7-3_14.png)   

  ![](/images/images_2018/7-3_15.png)   

  ![](/images/images_2018/7-3_16.png)  

### 迭代  

  ![](/images/images_2018/7-3_17.png)     

  ![](/images/images_2018/7-3_18.png)      

  ![](/images/images_2018/7-3_19.png)       

正向传播算法计算A2和代价函数：       

  ![](/images/images_2018/7-3_20.png)    

  ![](/images/images_2018/7-3_21.png)     

反向传播算法计算偏导数：  

   ![](/images/images_2018/7-3_22.png)    

注意np.dot()、np.sum()、np.multiply()和*的区别。   

   ![](/images/images_2018/7-3_23.jpg)    

   ![](/images/images_2018/7-3_24.png)    

使用梯度下降算法更新参数：  

   ![](/images/images_2018/7-3_25.png)      

   ![](/images/images_2018/7-3_26.png)      

测试结果：      

   ![](/images/images_2018/7-3_27.png)   

### 集成   

将之前的函数按照正确的顺序集成在一起，即nn_model()。   

   ![](/images/images_2018/7-3_28.png)    

参数初始化：     

   ![](/images/images_2018/7-3_29.png)    

迭代循环，循环里包括：正向传播算法计算A2和代价函数，反向传播算法计算偏导数，使用梯度下降算法更新参数：    

   ![](/images/images_2018/7-3_30.png)    

使用验证数据得到迭代后的代价函数值：  

   ![](/images/images_2018/7-3_31.png)    

### 预测   

使用正向传播算法预测y值，然后使用0.5作为阀值将y值转换为0或1。  

   ![](/images/images_2018/7-3_32.png)     

使用验证数据得到预估的平均准确度为0.67。 

   ![](/images/images_2018/7-3_33.png) 

运行模型查看分类效果，从图中可以看到神经网络模型可以比较好得完成分类任务：    

   ![](/images/images_2018/7-3_34.png)    

   ![](/images/images_2018/7-3_35.png)  

其分类准确度达到90%，远远高于逻辑回归模型的准确度。神经网络能够学习高度非线性的决策边界。    

   ![](/images/images_2018/7-3_36.png)  




 
     

