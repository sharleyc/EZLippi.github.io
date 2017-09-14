---
layout:     post
title:      如何进行无线adb调试
keywords:   博客
categories: [Android]
tags:	    [adb]
---

ADB，即Android Debug Bridge，它有两种调试方式USB或WIFI方式。本文介绍了如何开启android手机的adbd进行无线adb调试。

* 确保你的Android手机有root权限（检验方法输入su并回车，若命令提示符从$变#则为rooted）；确保PC和Android手机在同一网段下。开启手机的USB调试，用USB将Android手机连接到PC上。

* 进入Android SDK的platform-tools目录下，执行以下命令重启Android手机中的adbd后台程序重新侦听TCP的指定端口：adb tcpip 5555（成功则返回：restarting in TCP mode port:5555）

* 执行以下命令建立无线调试连接：adb connect <address>（address是手机的IP地址，成功则返回：connected to address:port），如果出现“unable to connect to address:port...”的信息则：

  +  在手机上下载一个shell软件，切换为root用户(su)。

  +  设置adbd服务监听的端口号：setprop service.adb.tcp.port 5555
 
  +  关闭服务：stop adbd 
 
 + 重启服务：start adbd
 
 + 进入platform-tools目录下执行：adb connect <address>




