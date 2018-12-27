---
layout:     post
title:      YOLO系列(一)
keywords:   博客
categories: [机器学习]
tags:	    [深度学习，物体检测，YOLO]
---

这是yolo算法系列的第一篇文章。本文主要记录在windows10中搭建环境的过程，和中间遇到的问题。

## 环境搭建

系统配置
OS： Windows, 64bit   
显卡：NVIDIA GeForce GTX 1060

已安装软件（环境变量均已配置好）：    
Python：3.6.6     
CUDA：9.0    
cuDNN：7.0.5     
Visual Studio Community 2017     

### 下载darknet

打开git bash，输入以下命令，下载darknet框架：

 - git clone https://github.com/AlexeyAB/darknet.git

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

https://github.com/AlexeyAB/darknet#how-to-compile-on-windows  
https://blog.csdn.net/u011635764/article/details/50564259



