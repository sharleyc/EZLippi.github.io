---
layout:     post
title:      常用adb命令以及使用问题汇总
keywords:   博客
categories: [Android]
tags:	    [adb]
---

本文汇总了常用的adb命令，以及笔者在使用adb时遇到的一些问题的解决方法。

## adb常用命令

* adb devices  
获取模拟器/设备列表 
    
* adb shell  
进入设备或模拟器的shell
 
* su  
切换到root权限

* exit  
退出root权限/shell环境


## adb使用问题

* 多device的时候，指定device来执行adb的方法
 adb后面用参数-s指定device： adb -s <devicename> shell
 
* 5037端口被占用导致adb无法正常使用   

    netstat -ano|findstr "5037"   
   -a 显示所有连接和监听端口   
   -n 以数字形式显示地址和端口号    
   -o 显示与每个连接相关的所属进程 ID  
   假设结果如下，PID为15032的进程占用了5037端口
    TCP    127.0.0.1:5037         0.0.0.0:0              LISTENING       15032  
	taskkill /F /PID 15032   
   /F是强制终止   
   /PID 或 /IM 表示根据进程ID还是进程名来区分进程
  
 
