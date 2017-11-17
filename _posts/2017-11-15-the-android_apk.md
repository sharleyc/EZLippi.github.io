---
layout:     post
title:      查看Android应用包名和Activity的方法
keywords:   博客
categories: [Android]
tags:	    [包名]
---

##  应用已经安装到设备上（没有APK包）    

使用命令： adb shell dumpsys window | findstr mCurrentFocus    
前提：启动应用    

## 常用APP的包名和Activity列表   

JOOX：   
com.tencent.ibg.joox/com.tencent.wemusic.ui.main.MainTabActivity    

VOOV：   
com.tencent.livemaster/com.tencent.livemaster.business.home.ui.HomeActivity   

微信：   
com.tencent.mm/com.tencent.mm.ui.LauncherUI   

QQ音乐：   
com.tencent.qqmusic/com.tencent.qqmusic.activity.AppStarterActivity    








