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

<center> **Figure 1** : **Definition of a box**</center>


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

首先，你需要了解模型的输入和输出是什么。输入是一批样本图片 (m, 608, 608, 3)，其中m是样本数；输出是边界框及其分类结果，其中c既可以是1-80中的任一数字，也可以是一个长度为80的向量，如果把c看作是向量，则一个边界框可以用85个数字来表示。因此这个深度卷积神经网络的输出是(m,19,19,5,85)。

We will use 5 anchor boxes. So you can think of the YOLO architecture as the following: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).

Lets look in greater detail at what this encoding represents. 

   ![](/images/images_2018/architecture.png)


<center>  **Figure 2** : **Encoding architecture for YOLO** </center>

If the center/midpoint of an object falls into a grid cell, that grid cell is responsible for detecting that object.

Since we are using 5 anchor boxes, each of the 19 x19 cells thus encodes information about 5 boxes. Anchor boxes are defined only by their width and height.

For simplicity, we will flatten the last two last dimensions of the shape (19, 19, 5, 85) encoding. So the output of the Deep CNN is (19, 19, 425).

第二步，为了简化计算，将最后两维的数据降为一维，因此DEEP CNN的输出为(19, 19, 425)。

   ![](/images/images_2018/flatten.png)

<center>  **Figure 3** : **Flattening the last two last dimensions** </center>

Now, for each box (of each cell) we will compute the following elementwise product and extract a probability that the box contains a certain class.

第三步，做点乘运算，并取最大值为分类结果。

   ![](/images/images_2018/probability_extraction.png)

<center>  **Figure 4** : **Find the class detected by each box**</center>

Here's one way to visualize what YOLO is predicting on an image:

- For each of the 19x19 grid cells, find the maximum of the probability scores (taking a max across both the 5 anchor boxes and across different classes). 
- Color that grid cell according to what object that grid cell considers the most likely.

Doing this results in this picture: 

通过单元格填色的方式，可视化了YOLO算法的预测结果。


   ![](/images/images_2018/proba_map.png)

<center>  **Figure 5** : Each of the 19x19 grid cells colored according to which class has the largest predicted probability in that cell.</center>


Another way to visualize YOLO's output is to plot the bounding boxes that it outputs. Doing that results in a visualization like this: 

通过画出边界框的方式，可视化了YOLO算法的预测结果。

   ![](/images/images_2018/anchor_map.png)

<center>  **Figure 6** : Each cell gives you 5 boxes. In total, the model predicts: 19x19x5 = 1805 boxes just by looking once at the image (one forward pass through the network)! Different colors denote different classes. </center>

In the figure above, we plotted only boxes that the model had assigned a high probability to, but this is still too many boxes. You'd like to filter the algorithm's output down to a much smaller number of detected objects. To do so, you'll use non-max suppression. Specifically, you'll carry out these steps: 

- Get rid of boxes with a low score (meaning, the box is not very confident about detecting a class)
- Select only one box when several boxes overlap with each other and detect the same object.


最后，使用non-max suppression算法过滤前面已经检测到的物体：舍弃低分数的boxes，以及保留重叠boxes里的最高分数的那个box。

### 2.2 - Filtering with a threshold on class scores

You are going to apply a first filter by thresholding. You would like to get rid of any box for which the class "score" is less than a chosen threshold. 

The model gives you a total of 19x19x5x85 numbers, with each box described by 85 numbers. It'll be convenient to rearrange the (19,19,5,85) (or (19,19,425)) dimensional tensor into the following variables:  
- `box_confidence`: tensor of shape $(19 \times 19, 5, 1)$ containing $p_c$ (confidence probability that there's some object) for each of the 5 boxes predicted in each of the 19x19 cells.
- `boxes`: tensor of shape $(19 \times 19, 5, 4)$ containing $(b_x, b_y, b_h, b_w)$ for each of the 5 boxes per cell.
- `box_class_probs`: tensor of shape $(19 \times 19, 5, 80)$ containing the detection probabilities $(c_1, c_2, ... c_{80})$ for each of the 80 classes for each of the 5 boxes per cell.

**Exercise**: Implement `yolo_filter_boxes()`.

1. Compute box scores by doing the elementwise product as described in Figure 4. The following code may help you choose the right operator: 
```python
a = np.random.randn(19*19, 5, 1)
b = np.random.randn(19*19, 5, 80)
c = a * b # shape of c will be (19*19, 5, 80)
```
2. For each box, find:
    - the index of the class with the maximum box score ([Hint](https://keras.io/backend/#argmax)) (Be careful with what axis you choose; consider using axis=-1)
    - the corresponding box score ([Hint](https://keras.io/backend/#max)) (Be careful with what axis you choose; consider using axis=-1)
3. Create a mask by using a threshold. As a reminder: `([0.9, 0.3, 0.4, 0.5, 0.1] < 0.4)` returns: `[False, True, False, False, True]`. The mask should be True for the boxes you want to keep. 
4. Use TensorFlow to apply the mask to box_class_scores, boxes and box_classes to filter out the boxes we don't want. You should be left with just the subset of boxes you want to keep. ([Hint](https://www.tensorflow.org/api_docs/python/tf/boolean_mask))

Reminder: to call a Keras function, you should use `K.function(...)`.


	# GRADED FUNCTION: yolo_filter_boxes
	
	def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
	    """Filters YOLO boxes by thresholding on object and class confidence.
	    
	    Arguments:
	    box_confidence -- tensor of shape (19, 19, 5, 1)
	    boxes -- tensor of shape (19, 19, 5, 4)
	    box_class_probs -- tensor of shape (19, 19, 5, 80)
	    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
	    
	    Returns:
	    scores -- tensor of shape (None,), containing the class probability score for selected boxes
	    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
	    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
	    
	    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
	    For example, the actual output size of scores would be (10,) if there are 10 boxes.
	    """
	    
	    # Step 1: Compute box scores
	    ### START CODE HERE ### (≈ 1 line)
	    box_scores = box_confidence * box_class_probs
	    ### END CODE HERE ###
	    
	    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
	    ### START CODE HERE ### (≈ 2 lines)
	    box_classes = K.argmax(box_scores,axis=-1)
	    box_class_scores = K.max(box_scores,axis=-1,keepdims = False)
	    ### END CODE HERE ###
	    
	    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
	    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
	    ### START CODE HERE ### (≈ 1 line)
	    filtering_mask = box_class_scores >= threshold
	    ### END CODE HERE ###
	    
	    # Step 4: Apply the mask to scores, boxes and classes
	    ### START CODE HERE ### (≈ 3 lines)
	    scores = tf.boolean_mask(box_class_scores ,filtering_mask)
	    boxes = tf.boolean_mask(boxes ,filtering_mask)
	    classes =  tf.boolean_mask(box_classes ,filtering_mask)
	    ### END CODE HERE ###
	    
	    return scores, boxes, classes

    #test code
	with tf.Session() as test_a:
	    box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
	    boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
	    box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
	    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
	    print("scores[2] = " + str(scores[2].eval()))
	    print("boxes[2] = " + str(boxes[2].eval()))
	    print("classes[2] = " + str(classes[2].eval()))
	    print("scores.shape = " + str(scores.shape))
	    print("boxes.shape = " + str(boxes.shape))
	    print("classes.shape = " + str(classes.shape))

执行结果：

scores[2] = 10.7506    
boxes[2] = [ 8.42653275  3.27136683 -0.5313437  -4.94137383]     
classes[2] = 7    
scores.shape = (?,)    
boxes.shape = (?, 4)    
classes.shape = (?,)   


### 2.3 - Non-max suppression ###

Even after filtering by thresholding over the classes scores, you still end up a lot of overlapping boxes. A second filter for selecting the right boxes is called non-maximum suppression (NMS).  


   ![](/images/images_2018/non-max-suppression.png)

<center>  **Figure 7**: In this example, the model has predicted 3 cars, but it's actually 3 predictions of the same car. Running non-max suppression (NMS) will select only the most accurate (highest probabiliy) one of the 3 boxes.  </center>


Non-max suppression uses the very important function called **"Intersection over Union"**, or IoU.

   ![](/images/images_2018/iou.png)

<center> **Figure 8** : Definition of "Intersection over Union". </center>

**Exercise**: Implement iou(). Some hints: 

- In this exercise only, we define a box using its two corners (upper left and lower right): `(x1, y1, x2, y2)` rather than the midpoint and height/width.
- To calculate the area of a rectangle you need to multiply its height `(y2 - y1)` by its width `(x2 - x1)`.
- You'll also need to find the coordinates `(xi1, yi1, xi2, yi2)` of the intersection of two boxes. Remember that:
    - xi1 = maximum of the x1 coordinates of the two boxes
    - yi1 = maximum of the y1 coordinates of the two boxes
    - xi2 = minimum of the x2 coordinates of the two boxes
    - yi2 = minimum of the y2 coordinates of the two boxes
- In order to compute the intersection area, you need to make sure the height and width of the intersection are positive, otherwise the intersection area should be zero. Use `max(height, 0)` and `max(width, 0)`.

In this code, we use the convention that (0,0) is the top-left corner of an image, (1,0) is the upper-right corner, and (1,1) the lower-right corner. 

这部分的编程作业里需要注意的是intersection区域的计算方法，需要用到提示里说的max方法，当没有交集区域时，应该确保结果为0，而不是负数。


	# GRADED FUNCTION: iou
	
	def iou(box1, box2):
	    """Implement the intersection over union (IoU) between box1 and box2
	    
	    Arguments:
	    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
	    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
	    """
	
	    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
	    ### START CODE HERE ### (≈ 5 lines)
	    xi1 = np.maximum(box1[0], box2[0])
	    yi1 = np.maximum(box1[1], box2[1])
	    xi2 = np.minimum(box1[2], box2[2])
	    yi2 = np.minimum(box1[3], box2[3])
	    inter_area = np.multiply(max((yi2 -  yi1),0) , max((xi2 - xi1),0))
	    ### END CODE HERE ###    
	
	    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
	    ### START CODE HERE ### (≈ 3 lines)
	    box1_area = np.multiply((box1[3] - box1[1]) , (box1[2] - box1[0]))
	    box2_area = np.multiply((box2[3] - box2[1]) , (box2[2] - box2[0]))
	    union_area = box1_area + box2_area - inter_area
	    ### END CODE HERE ###
	    
	    # compute the IoU
	    ### START CODE HERE ### (≈ 1 line)
	    iou = inter_area / union_area
	    ### END CODE HERE ###
	    
	    return iou

    #test code
	box1 = (2, 1, 4, 3)
	box2 = (1, 2, 3, 4) 
	print("iou = " + str(iou(box1, box2)))

执行结果：

iou = 0.142857142857


**Exercise**: Implement yolo_non_max_suppression() using TensorFlow. TensorFlow has two built-in functions that are used to implement non-max suppression (so you don't actually need to use your `iou()` implementation):

- [tf.image.non_max_suppression()](https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression)
- [K.gather()](https://www.tensorflow.org/api_docs/python/tf/gather)