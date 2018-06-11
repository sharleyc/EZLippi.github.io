---
layout:     post
title:      iOS性能数据自动化获取之内存篇  
keywords:   博客
categories: [性能测试]
tags:	    [iOS，性能测试]
---

APP引入以下API进行二次开发：    
https://github.com/andrealufino/ALSystemUtilities/tree/develop/ALSystemUtilities/ALSystemUtilities   

  ![](/images/images_2018/6-11_1.png)     

优点是：一次获取所有相关性能数据，极大提高专项测试效率；手机无需root；扩展性强，因为是iOS底层API二次开发，APP均可集成使用，集成简单方便；结合自动化脚本执行，可以实现测试自动化闭环，无需手工参与；无门槛，外包同学可使用。数据的准确性有保障（与instrument对比），开发认可。     


## iOS内存概念        

1、total Memory：设备总内存。   
2、Free Memory：可用内存，未使用的RAM容量，随时可以被应用分配使用。   
3、Used Memory：已使用内存，又可分为以下三种   
（1）Wired Memory：用来存放内核代码和数据结构，主要为内核服务，如负责网络、文件系统之类的。应用级别的软件是无法分配此内存的，但会影响其分配。（如何影响？）    
（2）Active Memory：活跃的内存，正在被使用或很短时间内被使用过。   
（3）Inactive Memory：非活跃的内存，最近被使用过，但是目前处于不活跃状态。例如，使用了邮件然后退出，则邮件曾经使用的RAM被标记为“不活跃”内存。不活跃内存可供其它应用软件使用，就像可用内存一样。   
（4）Purgeable Memory：可以理解为可释放的内存，主要是大对象或大内存块才可以使用的内存，此内存会在内存紧张的时候自动释放掉。    
4、AppUsedMemory：APP占用内存。    

## 低内存导致应用被杀问题的分析   
  
说明：  
当Free Memory低于某个值时（由物理内存大小决定），系统会按照以下顺序使用Inactive的资源：    
（1）首先，如果Inactive的数据最近被调用了，系统会把它们的状态改变成Active,并且在原有Active内存逻辑地址的后面。   
（2）其次，如果Inactive的内存数据最近没有被使用过，但是曾经被更改过，而还没有在硬盘的相应虚拟[内存]中做修改，系统会对相应硬盘的虚拟内存做修改，并把这部分物理内存释放为free供程序使用。  
（3）再次，如果inactive内存数据被在映射到硬盘后再没有被更改过，则直接释放成free。   
（4）最后如果active的内存一段时间没有被使用，会被暂时改变状态为inactive。
    
所以，如果你的系统里有少量的free memory和大量的inactive的memory，说明你的内存是够用的，系统运行在最佳状态，只要需要,系统就会使用它们，不用担心。   

如果系统的free memory和inactive memory都很少，而active memory很多，说明你的内存不够了。当然一开机，大部分内存都是free,这时系统反而不在最佳状态，因为很多数据都需要从硬盘调用，速度反而慢了。     

另一个角度理解iOS内存使用：   

用户使用的内存又分为四种：固定内存（Wired Memory），活跃内存（Active Memory），空闲内存（Inactive Memory），可用内存（Free Memory）。一般的释放内存软件大多是释放活跃内存，从而增加自由内存的可用量。 

## 产品应用   

以直播类产品为例，性能测试覆盖三个常见用户场景：主播开播、观看直播、观看回放，每个场景执行时间1小时，按照指定频率（比如1s）采集性能数据到文件中，测试结束后观察性能数据曲线图，重点关注APP内存以及系统内存。     

  ![](/images/images_2018/6-11_2.png)   

以上图为例，APP内存有小幅度增加，因为有少许内存泄漏；APP内存中间突然降低，因为产生了内存告警，APP自身处理告警的机制是清除缓存，比如清除放在共享内存中的弹幕；系统内存有几次内存的突增，对应的正是发送豪华礼物的时间点，由于礼物使用系统webview，计算到系统内存里了。发现异常到定位问题，需要结合日志记录系统（了解当时的操作）和工具（Intruments）分析定位到问题的代码行。        

      
扩展阅读：   
iOS 25个性能优化/内存优化常用方法     
https://www.2cto.com/kf/201505/401059.html     

参考文章：   
https://www.jianshu.com/p/fcbb9a472633   
https://blog.csdn.net/pizi0475/article/details/53423698   
 
