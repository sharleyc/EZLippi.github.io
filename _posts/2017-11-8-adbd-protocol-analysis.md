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

### 执行关联的命令  

我们通过上面的分析，了解到一个简单shell命令的执行过程。实际上，我们可能需要执行几个命令来完成一个任务，比如cd到某个路径下执行ls命令。我们仍然可以借助wireshark来分析关联命令的执行过程，这里就不详述了。需要注意的是，有关联的命令执行过程中，local-id和remote-id是固定的，而OPEN消息会改变接收方的ID，OPEN一次相当于生成一个新的stream。

## SYNC服务  
SYNC用于启动文件同步服务，用于实现“adb push”和“adb pull”。 由于此服务非常复杂，因此是单独在SYNC.TXT文档中详细介绍的(其它命令是在protocol.txt中介绍的)。    

使用协议请求同步服务（“sync：”）设置连接为同步模式。 这种模式是一种二进制模式，与常规adb协议不同。 在发送初始的“sync：”命令后，服务器必须用“OKAY”或“FAIL”这两个命令进行响应。     

在同步模式下，服务器和客户端会经常使用八字节的数据包来作为同步请求和同步响应。 前四个字节是一个id，指定同步请求是由四个utf-8字符组成。 最后四个字节是小端整数，代表各种用途。 后续用“长度”指代。 事实上所有二进制整数都是同步模式下的Little-Endian。 每个同步请求之后，隐式退出同步模式，并进行正常的adb通信。    

有四种同步请求命令字：LIST、RECV、SEND和STAT。    

以上信息来源于官方文档的描述。 下面我们就adb pull命令来理解该服务。  

第一步，查看TCP流，了解整体流程：  
 ![](/images/images_2017/pull_1.jpg)  
第二步，分析数据包序列，画一个简单的序列图：  
 ![](/images/images_2017/pull_2.jpg)   
第三步，对比官方文档和实际抓包数据    
### 请求同步服务("sync:")  
 ![](/images/images_2017/pull_3.jpg)  
OPEN的"destination"："sync:\x00"
请求成功，则收到OKAY包  
### 发送STAT同步请求  
 ![](/images/images_2017/pull_4.jpg)    
WRTE的data：STAT + length + str_path  
这个命令用于获取文件信息，请求成功，则收到OKAY回包  
### 收到STAT回复请求   
 ![](/images/images_2017/pull_5.jpg) 
WRTE的data：STAT + mode + size + time  
其中size是文件的大小。  
回复对方OKAY包   
### 发送RECV同步请求 
 ![](/images/images_2017/pull_6.jpg)  
WRTE的data：RECV + length + str_path 
这个命令用于将设备上的文件拷贝到本地。请求成功，则收到OKAY回包  
### 收到数据  
 ![](/images/images_2017/pull_7.jpg)  
WRTE的data：DATA + chunk_size 
实际文件以块形式发送。 每个块都遵循一定的格式。“DATA”后跟块大小字节数。每个文件块不得大于64k。 
每收到一个WRTE消息，需要回复一个OKAY消息。  
### 收到同步请求DONE  
 ![](/images/images_2017/pull_8.jpg)     
传送文件完成后，WRTE消息的数据最后会包含同步请求“DONE”，需回复OKAY响应该同步请求。   


  

