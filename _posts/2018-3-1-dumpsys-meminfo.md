---
layout:     post
title:      由浅入深聊Android--内存篇（一）
keywords:   博客
categories: [Android]
tags:	    [内存]
---

内存是Android APP的一个非常重要的性能指标之一。一旦分配出去的内存没有及时回收，会造成卡顿、程序变慢甚至OOM等问题，我们称之为内存泄漏(Memory leak)。简单的APP内存测试一般有两种场景，一种是静置APP，一种是重复进行同一个操作，通过绘制PSS曲线，观察内存是否有持续增长的现象。本文就来聊聊为什么内存衡量指标要选择PSS以及如何获取PSS。

## 内存衡量指标

常用的衡量指标有USS,PSS,RSS和VSS，在实际项目中一般选择PSS作为衡量指标。


* VSS是一个进程总共可访问的地址空间，其大小还包括了可能不在RAM中的内存（比如虽然malloc分配了空间，但尚未写入）。VSS很少被用于判断一个进程的真实内存使用量。      
* RSS是一个进程在RAM中真实存储的总内存，我们发现把所有RSS加起来会比全部实存RAM都大，原因是共享库被重复计算了。所以RSS并不能准确反映单进程的内存占用情况。为了更加合理的衡量每个进程的内存使用大小，引入了PSS。 
* PSS是进程消耗的非共享内存和按比例使用的共享内存部分。全部PSS加起来就是物理内存当前所占用的大小。当一个进程被销毁后，其占用的共享内存部分会再次按比例分配给余下使用该共享内存的进程。因此PSS不能准确表示返回给全局系统的内存。**因为PSS就是用户在手机设置里能看到的进程占用内存大小的指标（老板也能看到），所以非常重要，要特别关注**。
*  USS是一个进程所占用的私有内存，即该进程独占的内存。USS反映了运行一个特定进程真实的边际成本（增量成本），当一个进程被销毁后，USS是真实返回给系统的内存。

## adb shell dumpsys meminfo解读
Android本身提供了非常多用于测量内存的工具：free, showmap, procrank等，对于PSS，最方便易用的工具是dumpsys meminfo。示例：
adb shell dumpsys meminfo --package com.tencent.mm，在5.x的系统上查看meminfo信息如下：   
  ![](/images/images_2018/meminfo_red.jpg)   
红色框中的部分表示的是指定进程当前内存映射统计情况，有四列，分别是Pss Total, Private Dirty, Private Clean, Swapped Dirty。Pss前面讲了。要理解其它字段的含义，需要首先了解私有内存（Dirty and Clean）。    
* 私有内存，顾名思义是进程独占内存，也就是进程销毁时可以回收的内存容量。通常Private Dirty内存是最重要的部分，因为只被自己进程使用。Dirty内存是已经被修改的内存页，当发生换页时要写回磁盘。一般手机没有SWAP，**所以Private Dirty基本上常驻内存，需要关注。**Private Clean是私有的代码段，例如：应用自己的dex文件内存映射。   
*  Swapped Dirty在4.x里还没有。Linux有一种Zram交换技术，与传统交换区实现的不同之处在于，传统交换区是将内存中的页交换到磁盘中暂时保存起来，发生缺页的时候，从磁盘中读取出来换入。而zram则是将内存页进行压缩，存放在内存中的特定区域（a compressed block device in RAM，简称zram），发生缺页的时候，进行解压缩后换入。这是采用时间换空间的一种技术，该技术目前在各类内存受限的嵌入式系统中，尤其是Android手机上广泛应用。Swapped Dirty很可能在ZRAM中。（这块我的了解非常有限）

## 总结
Android内存的场景测试方法的缺点是不方便定位问题，更进一步的测试方法是精准分析法，初始时dump得到hprof_0文件，操作一次后dump得到hprof_1，对比hprof_0和hprof_1得到增加对象列表list_1，再操作一次后dump得到hprof_2,对比得到hprof_1和hprof_2得到增加对象列表list_2，对比list_1和list_2(通过使用mat的histogram功能)，两个列表中都有的对象需要详细分析是否存在内存泄漏问题。



