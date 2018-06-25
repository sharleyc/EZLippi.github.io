---
layout:     post
title:      windows下解决python3和python2的共存问题 
keywords:   博客
categories: [Python]
tags:	    [python2，python3,windows]
---
使用python进行开发，有时候需要python2环境，有时候需要python3环境。同时安装两个版本的python，在需要的时候可以进行切换使用，是最好不过的了。本文是在windows7 64位系统下进行的。  

## 下载并安装python2和python3    
  

* 在[官网](https://www.python.org/)下载python安装包

* 先安装python3，再安装python2 

* 设置环境变量，分别添加D:\Python36，D:\Python36\Scripts和D:\Python27，D:\Python27\Scripts

## 修改exe名字   
      

* 找到安装目录，把python3中的python.exe和pythonw.exe的名称改为python3.exe和pythonw3.exe；把python2中的python.exe和pythonw.exe的名称改为python2.exe和pythonw2.exe  

 ![](/images/images_2018/6-25_0.png)     

* 运行cmd，输入python3即可运行python3版本；输入python2即可运行python2版本。  

 

## 设置pip    

重新安装两个版本的pip，使得两个python版本的pip能够共存 

* 运行命令：python3 -m pip install --upgrade pip --force-reinstall，显示重新安装成功。   

  ![](/images/images_2018/6-25_1.png)         

* 运行命令：python2 -m pip install --upgrade pip --force-reinstall，显示重新安装成功。   

* 通过pip3 -V和pip2 -V 查看两个版本的pip信息，以后只需运行pip3 install和pip2 install即可安装各自的python包。  

  ![](/images/images_2018/6-25_2.png)      
