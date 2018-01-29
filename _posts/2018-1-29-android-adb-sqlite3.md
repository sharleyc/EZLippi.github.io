---
layout:     post
title:      Android中如何使用命令行查看sqlite3
keywords:   博客
categories: [Android]
tags:	    [Android,sqlite3]
---
在Android应用程序开发中，有时可能会用到系统自带的数据库sqlite3。以下记录了如何采用命令行的方式去查看数据库和表。。    
   
## 第一步   
adb devices命令，确认手机连上adb。
   
     
## 第二步
adb shell和su命令，进行shell命令界面，并切换成root身份。  
  
    
## 第三步
cd /data/data和cd <应用程序包名>命令，进入到app目录。
   

## 第四步
cd databases和ls命令，查看目录下有两个数据库文件，一个是创建的BookStore.db，一个是BookStore.db-journal，这是为了让数据库能够支持事务而产生的临时日志文件。

## 第五步
sqlite3 BookStore.db命令打开数据库，.tables命令查看数据库表信息。

## 第六步
select * from Book;查看表详细信息。

## 总结
1、Android程序中，创建的数据库一般存放在/data/data/<应用程序包名>/databases的目录下。   
2、使用"sqlite3 <数据库名>"命令来选择操作某个数据库。   
3、使用".talbes"命令查看数据库中包含哪些表。输入查询的SQL语句可以查询某表中的数据，需要注意的是要以分号来结束查询语句的输入。