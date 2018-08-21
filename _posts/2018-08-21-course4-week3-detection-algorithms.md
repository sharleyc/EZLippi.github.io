---
layout:     post
title:      目标检测（测试题）
keywords:   博客
categories: [机器学习]
tags:	    [卷积神经网络，detection]
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

Suppose you are applying a sliding windows classifier (non-convolutional implementation). Increasing the stride would tend to increase accuracy, but decrease computational cost.

- True
- False


正确答案：False。滑动窗口增大，会导致准确度下降，运算成本下降。

------------------------------------------

## 第7题

In the YOLO algorithm, at training time, only one cell ---the one containing the center/midpoint of an object--- is responsible for detecting this object.

- True
- False

正确答案：True。

----------------------------------------

## 第8题

What is the IoU between these two boxes? The upper-left box is 2x2, and the lower-right box is 2x3. The overlapping region is 1x1.


   ![](/images/images_2018/8-21_04.png)


- 1/6 
- 1/9
- 1/10
- None of the above


正确答案：第2项。这题可能选错到第3项。明确IoU的定义：交集区域/合并集区域，交集区域的大小是1，合并集区域的大小是4+6-1=9，因此IoU的大小是1/9。

-----------------------------

## 第9题

Suppose you run non-max suppression on the predicted boxes above. The parameters you use for non-max suppression are that boxes with probability \leq≤ 0.4 are discarded, and the IoU threshold for deciding if two boxes overlap is 0.5. How many boxes will remain after non-max suppression?

   ![](/images/images_2018/8-21_05.png)

- 3
- 4
- 5
- 6
- 7

正确答案：第3项。根据<=0.4的boxes会被舍弃的规则，car 0.26的box被舍弃；根据IoU>=0.5的会只保留最高阈值的box的原则，car 0.62的box被舍弃。因此最后剩下5个boxes。

--------------------------------------

## 第10题

Suppose you are using YOLO on a 19x19 grid, on a detection problem with 20 classes, and with 5 anchor boxes. During training, for each image you will need to construct an output volume yy as the target value for the neural network; this corresponds to the last layer of the neural network. (yy may include some “?”, or “don’t cares”). What is the dimension of this output volume?

- 19x19x(25x20)
- 19x19x(5x20)
- 19x19x(5x25)
- 19x19x(20x25)

正确答案：第3项。每个anchor box需要19x19x25(1+4+20) 个输出单元来定位，5个anchor boxes则为 19x19x(5x25)。



