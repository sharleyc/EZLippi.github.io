---
layout:     post
title:      Android实用技巧集锦（一）
keywords:   博客
categories: [Android]
tags:	    [apk]
---

本文记录一些Android实用技巧，以备不时之需。

## 获取手机中的某个应用的apk包

第一步，找到应用的包名（如果知道包名，请跳过这一步）   
adb shell pm list packages   

第二步，找到apk的位置(这里以微信为例)   
adb shell pm path com.tencent.mm    
package:/data/app/com.tencent.mm-1/base.apk(返回结果)    

第三步，从手机adb pull到电脑上   
adb pull /data/app/com.tencent.mm-1/base.apk   

## 通过adb shell dumpsys命令获取当前应用的component  

UI自动化测试中，一般通过识别控件和操作控件来完成用例的执行，如果在这个过程中有非预期的弹窗出现，就会导致控件识别失败，如何解决这个问题呢？  

我们可以利用adb shell dumpsys命令获取当前应用的component，具体命令如下：   
adb shell dumpsys window windows | grep mCurrent   
mCurrentFocus=Window{13eea3c8 u0 com.tencent.mm/com.tencent.mm.ui.LauncherUI} （当前活动窗口是微信的启动页的返回结果）  

其中我们需要的部分是：com.tencent.mm/com.tencent.mm.ui.LauncherUI，在程序中可以利用正则表达式把所需的结果过滤出来。   

再来回答前面提的问题，把非预期的各种窗口信息放在一个白名单中，用一个单独的线程无限循环检测当前活动窗口，如果在白名单中则调用该窗口对应的函数（通常的操作是关闭该窗口）。   
 

## 待续



