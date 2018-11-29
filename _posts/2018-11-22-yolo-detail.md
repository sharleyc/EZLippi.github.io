---
layout:     post
title:      YOLO算法分析
keywords:   博客
categories: [机器学习]
tags:	    [超参数，调优]
---

YOLO算法是一种目标检测算法，全称是You Only Look Once: Unified, Real-Time Object Dection。它的特点是速度快，预测过程只需要进行一次CNN运算，可以提供端对端的实时预测。本文会学习如何应用YOLO算法进行物体的检测。

## 第一个filter的实现

第一个filter需要对YOLO算法的预测结果进行处理，去除掉分值低于阈值的boxes。

### K.argmax(x, axis=-1)
说明：在给定轴上求张量之最大元素下标。   
如果box_scores 张量的形状为(19,19,5,80)，5表示anchor box的数量，80表示物体的类别数  
那么，box_classes 张量的形状为(19,19,5)，表示检测的物体类别

    box_classes = K.argmax(box_scores,axis=-1)

### K.max(x, axis=None, keepdims=False)
说明：求张量中的最大值。  
那么，box_class_scores 张量的形状为(19,19,5)，表示检测的物体类别对应的分值 
 
    box_class_scores = K.max(box_scores,axis=-1,keepdims = False)

### tf.boolean_mask(a, b)
说明：对张量应用布尔掩码。将m维a矩阵仅保留与b中"True"元素同下标的部分，并将结果展开到m-1维。

    filtering_mask = box_class_scores >= threshold
    scores = tf.boolean_mask(box_class_scores ,filtering_mask)
    boxes = tf.boolean_mask(boxes ,filtering_mask)
    classes =  tf.boolean_mask(box_classes ,filtering_mask)


## 非极大值抑制的实现

目前的目标检测算法的问题之一是，或许对同一目标有多次检测，非极大值抑制可以确保对每个对象只得到一个检测。其实现步骤是：

 - 取检测到的最大的那个概率的方框，高亮
 - 接下来计算所有剩下的方框，以及上一步的最大概率方框的IOU值，如果大于设定的阈值，则表示方框检测的是同一物体，需要调暗方框来表明它们被抑制了。
 - 重复前面两步，直至所有的方框被高亮或调暗。
 
### tf.image.non_max_suppression()

	tf.image.non_max_suppression(
	    boxes,
	    scores,
	    max_output_size,
	    iou_threshold=0.5,
	    score_threshold=float('-inf'),
	    name=None
	)

说明：以分数降序来选择边界框的一个子集。 

参数：   

 - boxes : 形状为[num_boxes,4]的二维浮点型张量
 - scores：形状为[number_boxes]的一维浮点型张量
 - max_output_size：一个标量整数，表示通过非极大值抑制选择的框的最大数量
 - iou_threshold：一个标量浮点数，表示判断方框是否相对于IOU重叠太多的阈值
 - name：操作的名称，可选

返回：    

 - selected_indices：形状为[M]的一维整数Tensor，表示从box张量中选择的指数，其中M <= max_output_size


### K.gather(reference,indices)
说明：在给定的张量中搜索给定下标的向量。
参数： 

  - reference：被搜索的向量
  - indices：整数张量，要搜索元素的下标

返回： 

  - 一个与参数reference数据类型一致的张量

示例：



参考：      
https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression







   ![](/images/images_2018/11-20_01.png)

 - 最重要的：
   - 学习率α
 - 其次重要的：
   - 动量项，0.9是不错的默认值
   - Mini-batch大小
   - 隐藏单元数量
 - 第三重要的：
   - 网络层数
   - 学习率衰减
 - Adam优化算法采用默认参数：
   - β1: 0.9
   - β2: 0.999
   - epsilon: 10^(-8)



## 如何选择超参数组合

### 在网格中随机取样

   ![](/images/images_2018/11-20_02.png)
   
调参的真正困难在于，你需要事先知道哪一个超参数对于你的模型更重要。随机取样能更充分的为最重要的超参数尝试尽可能多的组合。(无论最重要的超参数是哪个)


### 区域定位的抽样方案

   ![](/images/images_2018/11-20_03.png)

在这个方案中，需要你能大体确定一个最优区域，即最理想的超参数来自于这个区域。然后你就可以在更小的方块内进行更密集的抽样。


## 选取合适的尺度

均匀随机抽样在某些情况下是合理的方案，但它并不对所有的超参数都适用。

### 学习率的方案

   ![](/images/images_2018/11-20_04.png)

假设你认为学习率的下限是0.0001，上限是1，画出0.0001~1的数轴，如果均匀随机抽样，90%的
样本值将落在0.1~1的范围内。更合理的方法，应该是对数尺度搜索，而不是线性尺度。用python实现就是：

     r=-4*np.random.rand()

r为-4~0的随机数，则alpha值在10^(-4)~10^0之间。
