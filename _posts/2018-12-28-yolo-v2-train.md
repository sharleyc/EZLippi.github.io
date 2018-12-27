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















### OpenCV安装

  ![](/images/images_2018/12-26_08.png)

1、在官网下载安装OpenCV 3.4.0版本，解压路径为：
D:\programs\opencv3.4，并添加系统环境变量：
D:\programs\opencv3.4\build\x64\vc14\bin

2、将\opencv3.4\build\x64\vc14\bin目录下的opencv_world340.dll 和opencv_ffmpeg340_64.dll拷贝到\build\darknet\x64目录下

3、由于安装的CUDA版本是9.0，与build\darknet\darknet.vcxproj中的CUDA版本不一致，需要修改成自己的CUDA版本号。

  ![](/images/images_2018/12-26_09.png)

需要注意的是CUDA必须在MSVS安装之后安装。

### Visual Studio安装

如果你没有安装VS，那么就安装VS2015吧，因为yolo项目是用VS2015写的。如果你和我一样已经安装了VS2017，那么就需要安装扩展。

安装扩展的方式是到微软官网下载安装程序，下载链接：
https://visualstudio.microsoft.com/zh-hans/vs/   

  ![](/images/images_2018/12-26_01.png)  
  ![](/images/images_2018/12-26_02.png)  

## 环境配置

1、用VS打开\build\darknet\darknet.sln，打开darknet属性页。

 - C/C++ -->常规--> 附加包含目录，添加以下路径：
 D:\programs\opencv3.4\build\include  

  ![](/images/images_2018/12-26_06.png)

 - 链接器 -->常规--> 附加库目录，添加以下路径：
 D:\programs\opencv3.4\build\x64\vc14\lib  

  ![](/images/images_2018/12-26_07.png)

 - 链接器 -->输入--> 附加依赖项，添加：
 opencv_world340.lib（这里添加的是Release模式的，如果是Debug模式，文件的结尾有d）

 - VC++目录-->常规-->包含目录，添加以下三条路径(具体根据自己解压路径进行修改)：
 D:\programs\opencv3.4\build\include   
 D:\programs\opencv3.4\build\include\opencv  
 D:\programs\opencv3.4\build\include\opencv2  

 - VC++目录-->常规-->库目录，添加如下路径：
 D:\programs\opencv3.4\build\x64\vc14\lib  

2、如下图设置 x64 和 Release 模式，生成darknet

  ![](/images/images_2018/12-26_10.png)

我遇到了下面这个问题：

  ![](/images/images_2018/12-26_11.png)

解决办法是，到目录C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\extras\visual_studio_integration\MSBuildExtensions下，将以下三个文件拷贝到VS对应的目录下C:\Program Files (x86)\MSBuild\Microsoft.Cpp\v4.0\v140\BuildCustomizations（请根据自己的路径进行修改）。

  ![](/images/images_2018/12-26_12.png)

重新生成darknet，成功之后在\build\darknet\x64目录下会生成darknet.exe。

  ![](/images/images_2018/12-26_13.png)


参考文章：

https://github.com/AlexeyAB/darknet/tree/47c7af1cea5bbdedf1184963355e6418cb8b1b4f#how-to-train-pascal-voc-data  

https://blog.csdn.net/u011635764/article/details/50564259



