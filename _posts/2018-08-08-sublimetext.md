---
layout:     post
title:      Windows下安装sublime text的python编译环境 
keywords:   博客
categories: [Python]
tags:	    [sublimetext2，python2，python3]
---
Sublime Text是一个代码编辑器，它是2008年1月开发出来的，最初被设计为一个具有丰富扩展功能的Vim。Sublime Text具有漂亮的用户界面和强大的功能，同时是一个跨平台的编辑器，同时支持Windows,Linux,Mac OS X等操作系统。 由于Sublime Text3没有有效的注册码，我安装了Sublime Text2版本。

## 下载并安装sublime text 2.0.2版本    
  
http://www.sublimetext.com/2


## 注册码

亲测以下注册码有效
      
----- BEGIN LICENSE -----    
Andrew Weber     
Single User License     
EA7E-855605     
813A03DD 5E4AD9E6 6C0EEB94 BC99798F     
942194A6 02396E98 E62C9979 4BB979FE     
91424C9D A45400BF F6747D88 2FB88078    
90F5CC94 1CDC92DC 8457107A F151657B    
1D22E383 A997F016 42397640 33F41CFC    
E1D0AE85 A0BBD039 0E9C8D55 E1B89D5D    
5CDB7036 E56DE1C0 EFCC0840 650CD3A6    
B98FC99C 8FAC73EE D2B95564 DF450523     
------ END LICENSE ------    

## 安装Package Control   

* 通过快捷键 ctrl+` 或者 View > Show Console 打开控制台，然后粘贴相应的 Python 代码并enter执行：
https://packagecontrol.io/installation#st2   
   
* 如果由于某种原因，无法使用代码安装，可以通过以下步骤手动安装：   
 - 点击Preferences > Browse Packages菜单，进入打开的目录的上层目录，然后再进入Installed Packages目录
 - 下载 Package Control.sublime-package 并复制到Installed Packages/目录     
 - 重启Sublime Text        

## 配置python3的编译环境 

* 通过Tools > Build System > New Build System...，生成一个空配置文件：   

   ![](/images/images_2018/8-8_01.jpg) 

添加如下内容：   
	{   
		"cmd": ["D:/programs/Python37/python3.exe", "-u", "$file"],
		"file_regex": "^[ ]*File \"(...*?)\", line ([0-9]*)",
		"selector": "source.python"   
	}   
配置代码需要修改python程序安装路径，注意用反斜杠。   
* 完成配置代码输入后点击保存，在弹出的保存界面中使用默认保存路径（这样配置才能生效）    
* 重新进入编译环境设置，就可以看到刚才添加的python3编译环境来，点击勾选即可。

   ![](/images/images_2018/8-8_02.png)

* 用同样的方法，可以新建一个python2的编译环境。 
* 运行一个程序测试一下（要先保存程序文件，才能运行），成功。

   ![](/images/images_2018/8-8_03.png)
