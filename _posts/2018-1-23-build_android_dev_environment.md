---
layout:     post
title:      搭建Android开发环境
keywords:   博客
categories: [Android]
tags:	    [搭建环境]
---
PC机重装系统了，需要重新搭建Android开发环境，这次出乎意料的顺利，但还是决定记录下整个过程，以备不时之需。去年没有坚持Android的学习，今年2018年决定重新开始。    
   
## JDK的配置   
    
1、下载安装JDK1.8，官网地址：
http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html   
2、配置环境变量   
新建系统变量JAVA_HOME，比如 D:\Java\jdk1.8   
编辑path变量:
;%JAVA_HOME%\bin;%JAVA_HOME%\jre\bin;   
新建系统变量CLASSPATH：
.;%JAVA_HOME%\lib;%JAVA_HOME%\lib\tools.jar       
3、检查是否配置成功，cmd窗口输入java -version    
   
     
## Android Studio的配置
   
1、下载安装最新稳定版本Android Studio，官网地址：  
http://www.android-studio.org/     
2、配置代理，开发网代理是dev-proxy.oa.com 8080端口   
3、自定义安装，设置Android sdk目录（SDK已经下载的情况）      
  
    
## 新建项目完成gradle构建工具的加载
   
1、创建一个demo项目，在proxy settings中确认应用代理进行Gradle的加载，设置http和https代理，点击OK应用代理2、下载更新android sdk，配置代理（同上）  
2、Android SDK环境变量配置   
编辑path变量，加入SDK中platform-tools和tools的目录路径   
  

## 遇到的问题  
1、adb devices无法识别真机，adb interface有黄色感叹号的问题  
解决方法： 
http://blog.csdn.net/chtnj/article/details/49718809