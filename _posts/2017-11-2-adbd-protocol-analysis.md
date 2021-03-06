---
layout:     post
title:      ADBD协议分析（一）
keywords:   博客
categories: [Android]
tags:	    [adbd, wireshark, python]
---

作为一名开发者或测试者，会经常用到ADB。ADB的全称是Android Debug Bridge，它是android sdk里的一个工具，用这个工具可以直接操作管理android模拟器或者真机，它还可以执行很多手机操作，比如运行shell命令、上传下载文件、安装卸载程序等。利用adb，我们可以达到远程控制手机的效果。通过学习和分析adb架构以及adbd协议，我们可以实现手机的脱机远程控制。

## ADB架构

借用下面两张图来简单说明一下ADB的架构。
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
 4 启动wireshark开始抓包 
 5 运行命令：adb connect 192.168.199.*(手机IP地址)，停止抓包，并保存抓包数据
 6 重新启动抓包，运行命令adb shell和ls，停止抓包，并保存抓包数据
设置过滤规则tcp.port == 5555 
 
* 追踪TCP流，理解adbd协议的工作流程

Wireshark分析功能中有一个追踪TCP流的功能，可以将客户端和服务器的数据排好顺序使之容易查看。使用方法是：单击一个TCP包，右键选择追踪流--TCP流，TCP流就会在一个单独的窗口中显示出来。窗口中的文字用两种颜色显示，其中红色表示从源地址到目标地址的流量，蓝色则用来表示相反方向的流量。

1 我们已经知道了adbd协议是基于TCP协议的，TCP协议通过三次握手来建立连接，因此首先找到三次握手的部分。(分析之前先设置过滤规则tcp.port == 5555)  
   ![](/images/images_2017/connect_2.jpg)  
2 追踪接下来的TCP流，可以清晰的看到adbd建立通信的整个过程。这里涉及到两个命令字：CNXN和AUTH。  
   ![](/images/images_2017/connect_3.jpg)  
3 继续查看后面的TCP流，对应的是打开一个数据流，adbd返回结果数据的过程，这个过程涉及到三个命令字：OPEN,OKAY,WRTE。  
   ![](/images/images_2017/connect_4.jpg)   
   
* 结合官方文档，分析ADBD协议数据包首部

1 CONNECT(version, maxdata, "system-identity-string")   
现在，version=0x01000000，maxdata=256*1024。  
maxdata规定了接受数据长度的最大值，但是一些旧版本的adb把maxdata硬编码为4096，因此，第一次发给设备的CNXN和AUTH包的数据长度不能超过4096。  
对于adb server来说，CNXN包必须是第一个发送的数据包。  

看看实际抓包情况是怎样的，首先看adb server发给adbd的CNXN包：  
 ![](/images/images_2017/connect_5.jpg)   
参数command: 434e584e，因为是小端模式（数据的高字节保存在内存的高地址中，而数据的低字节保存在内存的低地址中，与我们的阅读习惯不一致），所以实际是0x4e584e43，符合文档描述。  
参数arg0(version)：00000001，实际是0x01000000，符合文档描述。  
参数arg1(maxdata)：00000400，实际是0x00040000，即256*1024，符合文档描述。  
数据部分：'host::features=cmd,shell_v2'，systemtype有三种类型：bootloader、device、host，host表示adb server。  
  
再来看看adbd发给adb server的CNXN包：  
 ![](/images/images_2017/connect_6.jpg)   
参数command: 434e584e，实际是0x4e584e43，符合文档描述。  
参数arg0：00000001，实际是0x01000000，符合文档描述。  
参数arg1：00100000，实际是0x00001000，即4096，符合文档描述。  
数据部分：'device::ro.product.name=...'，device表示设备。  

2 AUTH(type, 0, "data")  
有些设备在连接建立之前需要验证。adbd收到adb server的CNXN包后，会发送一个type为TOKEN(1)的AUTH包，里面包含一个20字节的随机token。adb server用私钥给这个token进行签名并返回给adbd一个type为SIGNATURE(2)的AUTH包，adbd验证签名正确后，会响应adb server一个CNXN包，否则会向adb server发送新的AUTH包，提供新的随机TOKEN，adb server则尝试另一组公私钥对。如果尝试完所有的公私钥对，则向adbd发送AUTH RSAPUBLICKEY包，包含一个server公钥。adbd将该公钥发往framework，后者弹出信息框，询问是否允许使用USB调试接口，该信息框中一般会显示公钥指纹(MD5)，而不是公钥本身。   

下面我们还是根据实际抓包来了解验证过程。先看adbd发给adb server的AUTH包：   
 ![](/images/images_2017/auth_1.jpg)   
参数command: 41555448，因为是小端模式，所以实际是0x48545541，符合文档描述。  
参数arg0(type)：01000000，即1，符合文档描述。  
参数arg1(0)：00000000，即0，符合文档描述。  

再来看看adb server发给adbd的AUTH包：  
 ![](/images/images_2017/auth_2.jpg)   
参数command：0x48545541，符合文档描述。  
参数arg0(type)：2，符合文档描述。  
参数arg1(0)：0，符合文档描述。    

我们可以看到AUTH验证通过后，adbd随即响应了一个CNXN包，至此连接建立成功。
  

