---
layout:     post
title:      如何使用tcpdump和wireshark对Android手机进行抓包分析
keywords:   博客
categories: [Android]
tags:	    [抓包]
---

本文主要介绍如何使用tcpdump和wireshark对Android应用程序进行抓包分析。前提条件：Android设备已经root，并且PC上有Android SDK环境。

* 下载安装tcpdump。根据Android设备版本选择一个版本下载并解压其中的tcpdump文件，push到Android设备中。链接：http://www.tcpdump.org；命令： 
adb push c:\tcpdump /data/local/tcpdump

* 执行以下命令开始抓包：
/data/local/tcpdump -p -vv -s 0 -w /sdcard/capture.pcap

* 执行相应操作完毕后ctrl+c停止抓包
  
* 将capture.pcap文件pull到本地（比如d:/download目录下），使用wireshark进行分析：
adb pull /sdcard/capture.pcap d:/download






