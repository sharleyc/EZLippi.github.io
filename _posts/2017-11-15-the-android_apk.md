---
layout:     post
title:      获取Android应用包名和启动页Activity的方法
keywords:   博客
categories: [Android]
tags:	    [包名]
---

包名（Package name）在Android系统中是判断一个APP的唯一标识。Activity是APP中的页面，一个页面就是一个Activity，启动页Activity就是第一次启动起来的页面。我们在做自动化测试时，通常第一步是启动被测应用，启动方法就是利用“包名/启动页Activity”。以下是几种获取Android应用包名和启动页Activity的方法。   

##  看源代码   

有源代码的时候，我们可以非常容易的知道包名。打开工程目录下的AndroidManifest.xml文件，找到package=""这一项查看即可。包含category的值为"android.intent.category.LAUNCHER"的Activity即我们要找的启动页Activity。    

##  应用已经安装到设备上（没有APK包）    

使用命令：$ monkey -p [PackageName] -vvv 1    

## 启动速度获取方法   

adb shell am start -W [PackageName]/[PackageName.MainActivity]    

## 常用APP的包名和Activity列表   

JOOX：   
com.tencent.ibg.joox/com.tencent.wemusic.ui.main.LauncherUI   

VOOV：   
com.tencent.livemaster/com.tencent.livemaster.business.login.ui.LoginActivity   

微信：   
com.tencent.mm/com.tencent.mm.ui.LauncherUI   
 

QQ音乐：   
com.tencent.qqmusic/com.tencent.qqmusic.activity.AppStarterActivity    

参考资料：    
http://blog.csdn.net/wjw_java_android/article/details/52299353    








