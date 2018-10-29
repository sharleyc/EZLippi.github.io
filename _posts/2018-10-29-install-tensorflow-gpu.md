---
layout:     post
title:      Win10下Tensorflow GPU版安装回顾
keywords:   博客
categories: [机器学习]
tags:	    [Tensorflow, GPU, 安装]
---

我为什么要写这样一篇文章？方便其他人参考，顺利的完成Tensorflow GPU版的安装，同时也方便自己需要重装时的回顾。


# 前提条件

我的PC机OS和显卡配置如下：

OS: Windows10, 64bit    
显卡：NVIDIA GeForce GTX 1060

首先需要确认的是，显卡是否支持安装Tensorflow GPU版本。到以下网站查询你的显卡型号是否在列表中，如果不在，则不支持；如果在，请找出显卡的计算能力（Compute Capability），比如我的显卡对应的计算能力是6.1。   
https://developer.nvidia.com/cuda-gpus       
   ![](/images/images_2018/10-29_01.png)

请一定参考Tensorflow官网进行后续的安装。官网上有这样一段话：   

CUDA 计算能力为 3.0 或更高的 GPU 卡（用于从源代码编译），以及 CUDA 计算能力为 3.5 或更高的 GPU 卡（用于安装我们的二进制文件）

由于我的显卡计算能力是6.1，可以没有顾虑的开始安装了。

# 安装CUDA工具包

如果你打开CUDA官网，最新的CUDA工具包版本是10.0，但是Tensorflow官网上明确指出，必须在系统上安装CUDA 工具包9.0。建议大家直接下载安装9.0版本。
   ![](/images/images_2018/10-29_02.png)    

点击Legacy Releases，找到9.0版本。    

   ![](/images/images_2018/10-29_03.png)   

选择符合OS配置的版本进行下载，其中local版本是完整版。我下载安装了Base installer以及4个Patch包。  
 
   ![](/images/images_2018/10-29_04.png)   

在安装CUDA时提示我没有安装Visual Studio，按照提示安装即可。   
     
   ![](/images/images_2018/10-29_05.png) 

安装完成后，程序会自动添加两个系统变量：   

   ![](/images/images_2018/10-29_06.png) 

手动添加以下变量：  

CUDA_LIB_PATH = %CUDA_PATH%\lib\x64    
CUDA_BIN_PATH = %CUDA_PATH%\bin    

最后检查CUDA是否安装成功，进入以下目录：  

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\extras\demo_suite>  

分别输入deviceQuery.exe和bandwidthTest.exe，如果都显示Result = PASS，则说明安装成功了。   

   ![](/images/images_2018/10-29_07.png) 

还有一种验证方法是，输入nvcc -V，显示如下，也说明安装成功了。   

   ![](/images/images_2018/10-29_08.png)   

# 安装cuDNN  

Tensorflow官网上明确指出，必须在系统上安装cuDNN V7.0。因此我选择安装v7.0.5版本。   （https://developer.nvidia.com/rdp/cudnn-archive）  

   ![](/images/images_2018/10-29_09.png) 

cuDNN下载解压缩，把压缩包中bin,include,lib中的文件分别拷贝到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0目录下对应目录中； 把C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\extras\CUPTI\libx64\cupti64_80.dll 
拷贝到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin。  

# 安装Tensorflow GPU版本

官网中提到，在Windows上，Tensorflow支持Python3.5.x和3.6.x。我之前已经安装了Python3.6.6版本。因此直接输入以下命令：  

C:\> pip3 install --upgrade tensorflow-gpu  

需要注意的是，如果是在公司内网环境，可能需要配置代理和设置pip源。  

