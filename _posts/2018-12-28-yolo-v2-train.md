---
layout:     post
title:      YOLO系列之二：基于yolov2进行自定义训练
keywords:   博客
categories: [机器学习]
tags:	    [深度学习，物体检测，YOLO]
---

这是YOLO系列的第二篇文章。本文介绍如何基于YOLOv2用自己的数据进行训练并预测。

## YOLO介绍

YOLO是目前比较流行的目标检测(Object detection)算法，用它处理图像简单直接，而且非常快，因此可以实现实时检测，YOLOv3是截止到目前，最快的目标检测算法。本文介绍的是YOLOv2的训练和预测。

## YOLOv2如何进行训练

1、下载预训练权重文件，并把文件放在目录build\darknet\x64下

  http://pjreddie.com/media/files/darknet19_448.conv.23

  ![](/images/images_2018/12-28_01.png)

2、将要训练的数据放到目录build\darknet\x64\data下，这里文件夹的名字检测目标nfpa(这里的数据是从网上下载的，非自己标注)

  ![](/images/images_2018/12-28_02.png)

3、样本统一采用jpg格式的图像，标注文件的名字和样本的名字保持一致

  ![](/images/images_2018/12-28_03.png)

标注内容的格式为： [category number][object center in X][object center in Y][object width in X][object width in Y]

  ![](/images/images_2018/12-28_04.png)

将样本分为两部分，一部分作为训练集(train.txt)，一部分作为测试集(test.txt)，内容格式如下：

  ![](/images/images_2018/12-28_05.png)

4、在目录build\darknet\x64\data下添加obj.data和obj.names

obj.data的内容如下：   

  ![](/images/images_2018/12-28_06.png)

class表示检测的类别数，backup是训练过程中存放权重文件的目录

obj.names的内容如下：    

  ![](/images/images_2018/12-28_07.png) 

每个类别的名称占一行，这里只有一个类别   

5、在目录build\darknet\x64\cfg下拷贝yolo-voc.2.0.cfg重命名为yolo-obj.cfg，修改参数 

  ![](/images/images_2018/12-28_08.png) 

我们这里类别为1，因此设置class=1；filters=(class + 5) * 5，因此设置filters = 30   

6、打开cmd窗口，进入darknet.exe所在的目录，运行以下命令：  

    darknet.exe detector train data/obj.data cfg/yolo-obj.cfg darknet19_448.conv.23

7、训练过程中每隔1000个迭代，会在目录build\darknet\x64\backup下，自动生成一个权重文件   

  ![](/images/images_2018/12-28_09.png)  

8、将权重文件拷贝到darknet.exe所在的目录下，执行以下命令进行预测

    darknet.exe detector test data/obj.data cfg/yolo-obj.cfg yolo-obj_5000.weights data/nfpa/pos-24.jpg

  执行结果：

  ![](/images/images_2018/12-28_10.png)  

  ![](/images/images_2018/12-28_11.jpg)   

  

参考文章：

https://github.com/AlexeyAB/darknet/tree/47c7af1cea5bbdedf1184963355e6418cb8b1b4f#how-to-train-pascal-voc-data  

https://blog.csdn.net/u011635764/article/details/50564259



