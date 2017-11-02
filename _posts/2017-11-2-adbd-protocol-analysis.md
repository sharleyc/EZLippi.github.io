---
layout:     post
title:      ADBD协议分析
keywords:   博客
categories: [Android]
tags:	    [adbd, wireshark, python]
---

作为一名开发者或测试者，会经常用到ADB。ADB的全称是Android Debug Bridge，它是android sdk里的一个工具，用这个工具可以直接操作管理android模拟器或者真机，它还可以执行很多手机操作，比如运行shell命令、上传下载文件、安装卸载程序等。利用adb，我们可以达到远程控制手机的效果。通过学习和分析adb架构以及adbd协议，我们可以实现手机的脱机远程控制。

## ADB架构

借用下图来简单说明一下ADB的架构。
  ![](/images/images_2017/adb_1.jpg)
  ![](/images/images_2017/adb_2.jpg)

* adb client

adb client的作用是连接adb server，并将用户输入的指令转发给adb server，并等待adbd或adb server的应答。当输入adb指令比如adb devices时，实际上是在运行一个adb client，它会去连接adb server。

* adb server

adb server是作为一个后台服务在运行的，服务进程监听本地的5037端口，使用的是TCP协议。它会等待接收来自adb client的请求，并做相应的处理。这个进程在电脑上最多只有一个。

* adb daemon(adbd)

 运行于手机或模拟器的adbd进程。一些命令，adb client发送至adb server就结束了，但一些命令是adbd完成的，adb server只是充当了数据转发的功能，是adb client和adbd的桥梁。



## ADBD协议分析

前面介绍了adb的架构(adb client和adb server实际上是一个exe文件，仅仅是在启动的时候通过不同的命令行来区别的)。如果我们想脱机控制手机，需要模拟实现adb server和adbd的通信过程。

* ADBD协议的首部

根据官方文档的说明，adbd协议是二进制的，首部24个字节，包含6个字段，各占4个字节。     
 1 command： 命令字  
 2 arg0：第一个参数  
 3 arg1：第二个参数  
 4 data_length：数据部分的长度  
![](/images/images_2017/adbd_1.jpg)
![](/images/images_2017/adbd_2.jpg)

* 抓包分析

仅仅看官方说明或者源码，并不容易理解adbd协议的实现方式。这里要借助wireshark来抓包分析了。具体步骤如下：   
 1 手机USB连接PC，保证手机和PC在一个局域网中  
 2 运行命令：adb tcpip 5555  
 3 拔掉USB连接  
 4 运行命令：adb connect 192.168.199.*(手机IP地址)  
 5 启动wireshark开始抓包  
 6 运行命令adb shell ls  
 7 停止抓包，设置过滤规则tcp.port == 5555 
