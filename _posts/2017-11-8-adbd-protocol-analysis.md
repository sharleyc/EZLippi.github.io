---
layout:     post
title:      ADBD协议分析（二）
keywords:   博客
categories: [Android]
tags:	    [adbd, wireshark, python]
---

在上一篇文章里我们首先说明了ADB的架构，其中核心的三个部分是adb client，adb server，以及adb daemon。理论上，我们可以模拟实现adb server和adbd的通信过程，前提是了解adbd协议。通过wireshark抓包，以及官方文档的描述，我们分析了adbd协议建立连接的过程。在本文里我们将继续分析adbd协议的其它部分。

## 执行命令

首先来看看adbd协议是如何完成最常用的一个adb shell命令的。   
操作步骤：   
1、启动wireshark抓包  
2、在cmd窗口执行adb shell命令  
3、成功进入交互模式，停止抓包。   

### 分析抓包数据    
先看TCP流，红色是adb server发往adbd的，蓝色是反方向的，这里涉及到3个新的命令字OPEN,OKAY,WRTE：    
 ![](/images/images_2017/shell_1.jpg)  

#### OPEN(local-id, 0, "destination")  
OPEN消息告诉接收方，发送方有一个数据流需要连接到"destination"，其中local-id为本地ID，表示发送方，local-id不能为0。如果连接成功，会收到一个READY消息，否则会收到CLOSE消息。结合实际抓包数据看看OPEN消息的内容：  
 ![](/images/images_2017/shell_2.jpg)   
参数command: 4f50454e，因为是小端模式，所以实际是0x4e45504f，符合文档描述。  
参数arg0(local-id)：8，数据流不变，则代表协议一方的local-id是不会变的，这点可以在后面看到。  
参数arg1(0)：0，符合文档描述。这里其实是remote-id，因为还不知道remote-id，所以默认为0。  
数据部分（"destination"）："shell:"    

#### READY(local-id, remote-id, "")  
READY消息告诉接收方，数据流连接是正常的，可以发送消息。其中local-id是本地ID(发送方)，remote-id是远程ID(接收方)，数据部分为空。  
 ![](/images/images_2017/shell_3.jpg)  
参数command: 0x59414b4f，即OKAY，符合文档描述。  
参数arg0(local-id)：1，代表adbd。  
参数arg1(remote-id)：8，代表adb server。   

#### WRITE(local-id, remote-id, "data")  
WRITE消息发送数据给接收方，同READY消息一样，需要包含local-id和remote-id。并且只有接收到一条READY消息，才会发送一条WRITE消息。如果remote-id不正确，或者没有接收到READY消息，则stream会关闭。  
 ![](/images/images_2017/shell_4.jpg)    
参数command: 0x45545257，即WRTE，符合文档描述。  
参数arg0(local-id)：1，代表adbd。  
参数arg1(remote-id)：8，代表adb server。   
数据部分：shell@HWTAG-L6753:/$（回显在cmd窗口中的文字）   

继续执行ls，也是类似的过程，但不会有OPEN消息了。 画一个简单的序列图如下： 
 ![](/images/images_2017/shell_5.jpg)    

## 执行关联的命令  


  

