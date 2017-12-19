---
layout:     post
title:      adb使用问题汇总
keywords:   博客
categories: [Android]
tags:	    [adb]
---

在研究adb的过程中遇到一些坑，记录下来，避免重蹈覆辙。

## 问题一   

1、通过运行命令adb tcpip 5555把默认的USB模式改为TCP模式。

2、运行adb connect [device ip]，就可以在TCP模式下调用adb命令了

3、如果想调试脚本（模拟pc的adb server与设备的adbd通讯），则不应该执行步骤2  

4、未执行步骤2的情况下，查问题可以执行命令netstat -ano | findstr 5555，检查设备的5555端口是否被占用，如果被占用，则可以运行taskkill /pid [pid] /t /f杀掉占用该端口的进程。（/t tree kill 终止指定的进程和其子进程；/f 指定要强行终止的进程）   

## 问题二   

利用binascii模块，将二进制内容和ASCII进行转换。    

    from binascii import *   
    unhexlify('0d0a') #'\r\n' 回车换行符  
    hexlify('\r\n') # '0d0a'    
    
## 问题三   

两种拷贝手机文件到PC的方法：使用工具或者使用adb pull命令。而使用工具连接手机后，再使用adb命令会提示5037端口被占用，如果退出工具、杀死占用进程都无效的情况下，可以尝试重启电脑来解决问题。最后还是推荐使用adb命令，工具会卡，也不怎么好用。





  

