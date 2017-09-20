---
layout:     post
title:      抓包工具之RawCap
keywords:   博客
categories: [网络编程]
tags:	    [RawCap]
---

RawCap是针对windows的免费开源软件，能够抓取windows本地回环接口127.0.0.1 (localhost）的数据包。

## RawCap的使用

* RawCap.exe 127.0.0.1 dumpfile.pcap

* 交互式的方式：在命令行中直接输入RawCap.exe

* 抓好包后，按Ctrl+C停止抓包

## RawCap的局限性

* 无法抓取IPV6的数据包
* XP系统，只能抓取UDP和ICMP的数据包，不能抓取TCP包
* windows 7系统，无法捕获传入的数据包
* Windows Vista系统，无法捕捉传出的数据包