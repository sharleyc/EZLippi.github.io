---
layout:     post
title:      Python3和Python2的区别
keywords:   博客
categories: [Python]
tags:       [python3, python2]
---

Python3.*版本相对于Python的早期版本，是一个较大的升级，并且没有考虑向下相容。在实际使用过程中，新的Python程序会开始使用Python3.*版本的语法。Python3.*的变化主要在以下几个方面(持续更新)：

## print函数    

print语句没有了，只有print函数。  
    print "aa"  #Python2.6和Python2.7里支持，但Python3.*不支持
    print("aa") #print()不能带有任何其它参数


## Unicode      

在Python3的源码文件中默认使用utf-8编码(Python2是ascii码):   
    >>>imort sys   
    >>>sys.getdefaultencoding()
    'utf-8'     
在Pyhthon3中分别用str和bytes表示文本字符（总是Unicode）和二进制数据（Python2中把字符串分为unicode和str两种类型，容易误导开发者）。借用网上的一个图来世界str和bytes的转换关系：        

  ![](/images/images_2018/str-bytes.png)   

str可以编码成bytes，而bytes可以解码成字符串。    

## 除法运算    

Python的除法有两个运算符，/和//。/对于整数之间的相除，结果会是浮点数：    
    >>>1/2     
    0.5     
对于//除法(又叫floor除法)，会对结果自动进行一个floor操作(返回结果的下舍整数)：    
    >>> -1/2    
    -1    

##  比较操作符    

Python3中只有同一数据类型的对象可以比较   
    11 < 'test'  # TypeError: '<' not supported between instance of 'int' and 'str'    
    
##  range 与 xrange    

Python2中 range(0,4)  结果是[0,1,2,3]；python3中 改为list(range(0,4))   
Python2中 xrange(4) 适用于for循环的变量控制；python3中 改为 range(4)    

## 字符串    

Python2中，字符串以8-bit字符串存储；    
Python3中，字符串以16-bit Unicode字符串存储    

## 打开文件    

Python3中，只能用open(......)    

## try except语句   

改为try:    
        ......
    except Exception as e:
        ......    
        








