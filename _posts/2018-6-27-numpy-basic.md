---
layout:     post
title:      numpy基础入门学习 
keywords:   博客
categories: [Python]
tags:	    [numpy，python3]
---
Numpy(Numerical Python)是高性能科学计算和数据分析的基础包。最近在学习deep learning，向量化的能力是一个关键的技巧。运用numpy的内置函数，避免for循环，实现向量化可以有效地提高代码的运行速度。

## 常用方法     
  

### numpy.array     

numpy.array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0) 

numpy数组是一个多维数组对象，称为ndarray。使用array创建数组时，参数必须是由方括号括起来的列表，而不能使用多个数值作为参数调用array。 比如a = array(1,2,3)是错误的。  


	import numpy as np
	a = np.array([1,2,3])
	print(a) 

运行结果：

[1 2 3]


### numpy.zeros

numpy.zeros(shape, dtype=float, order='C')：返回给定形状和类型的新数组，用零填充。

	import numpy as np
	a = np.zeros(5)
	print(a)
	a = np.zeros(5,np.int8)
	print(a)  
	b = np.zeros((2,3))
	print(b)

运行结果： 

[0. 0. 0. 0. 0.]   
[0 0 0 0 0]   
[[0. 0. 0.]   
 [0. 0. 0.]]

默认情况下，zeros创建的数组是浮点型的，可以设置dtype参数为其它需要的类型（示例中为整型）。  

### numpy.dot   

numpy.dot(a, b, out=None)：计算两个数组a、b的乘积。注意不要和点积（*）搞混了，真正的点积是元素对应相乘。    

	import numpy as np
	a = np.array([1,2])
	b = np.array([3,4])
	print(np.dot(a,b))
	a = np.array([[1,2],[3,4]])
	b = np.array([[3,4],[5,6]])
	print(np.dot(a,b)) 

运行结果： 

11   
[[13 16]    
 [29 36]]

运用向量化运算来处理数组的速度，比借助循环语句完成相同的任务，运行效率要好得多，特别是运算量大的时候。   
 
	import time
	a = np.random.rand(1000000)
	b = np.random.rand(1000000)
	
	tic = time.time()
	c = np.dot(a,b)
	toc = time.time()
	
	print("Vertorized version:" + str(1000*(toc-tic)) + "ms")
	
	c=0 
	tic = time.time()
	for i in range(1000000):
	    c += a[i]*b[i]
	toc = time.time()
	
	print(c)
	print("For loop:" + str(1000*(toc-tic))+"ms") 

运行结果： 

Vertorized version:0.9999275207519531ms
250276.1277822864
For loop:480.9999465942383ms 

以上例子显示向量化版本仅为循环版本时间的1/480。


### numpy.exp 

numpy.exp(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'exp'>：返回e的n次方，e是一个常数为2.71828。  

	import numpy as np
	a = 1
	print(np.exp(a))
	a=2
	print(np.exp(a))
	print(2.71828182846*2.71828182846)  

运行结果： 

2.718281828459045   
7.38905609893065   
7.38905609893584   

### numpy.log   

numpy.log(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'log'>：自然对数（基于e）  

	import numpy as np
	print(np.log(np.e))
	print(np.log10(100))
	print(np.log2(8))  

运行结果：  

1.0   
2.0   
3.0   

以上示例是以e,10，2为底，如果要使用任意数为底就需要用到换底公式，比如以3为底4的对数，math.log(4,3)和np.log(4)/np.log(3)是等效的。    

### numpy.maximum  

numpy.maximum(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'maximum'>：x1和x2逐位比较取其大者。
  
	import numpy as np
	print(np.maximum([-2,-1,0,1,2],0)) 

运行结果：  

[0 0 0 1 2]  

以上示例中第二个参数只是一个单独的值，用到了broadcasting机制。  


### numpy.random.randn    

numpy.random.randn(d0, d1, ..., dn)：生成一个浮点数或N维浮点数组，取数范围：正态分布的随机样本数。注意结果不一定是[0,1)之间的随机数。

	import numpy as np
	a=np.random.randn(5)
	print(a)  

运行结果：  

[ 1.43773673 -0.15133186 -0.12978707  0.98659122  0.89018614]


### numpy.sum  

numpy.sum(a, axis=None, dtype=None, out=None, keepdims=<class 'numpy._globals._NoValue'>)[source]     

axis 为0时，将每一列的元素相加，将矩阵压缩为一行；axis 为1时，将每一行的元素相加，将矩阵压缩为一列；axis为None时，则将矩阵中的每一个元素都加起来。 

	import numpy as np
	A=np.array([[56.0,0.0,4.4,68.0],
	           [1.2,104.0,52.0,8.0],
	           [1.8,135.0,99.0,0.9]])
	cal = A.sum(axis=0)
	print(cal) 
	cal = A.sum(axis=1)
	print(cal) 
	cal = A.sum(axis=None)
	print(cal) 


运行结果：  

[ 59.  239.  155.4  76.9]  
[128.4 165.2 236.7]    
530.3000000000001
  
keepdims主要用于保留矩阵的二维特性。使用此选项，结果将正确地针对输入数组进行广播。

	import numpy as np
	A=np.array([[56.0,0.0,4.4,68.0],
	           [1.2,104.0,52.0,8.0],
	           [1.8,135.0,99.0,0.9]])
	cal = A.sum(axis=0, keepdims=True)
	print(cal)

运行结果： 

[[ 59.  239.  155.4  76.9]]

### numpy.ndarray.shape 

ndarray.shape：输出矩阵的维度，即行数和列数。shape[0]和shape[1]分别代表行和列的长度。 

	import numpy as np
	a=np.random.randn(5)
	print(a)
	print(a.shape)
	print(a.shape[0])
	b = np.random.randn(5,1)
	print(b)
	print(b.shape) 
	print(b.shape[0],b.shape[1])

运行结果：  

[ 0.24152085 -0.43457502  0.56065198  1.59229367 -0.13256503]  
(5,)   
5   
[[-0.10656198]
 [-0.33429877]
 [ 0.82345545]
 [ 2.90139723]
 [-0.09680888]]   
(5, 1)   
5 1

### numpy.reshape  

numpy.reshape(a, newshape, order=’C’):给矩阵一个新的形状而不改变其数据。  

	import numpy as np
	a=[1,2,3,4,5,6]
	print(np.reshape(a,(2,3))) 

运行结果：  

[[1 2 3]   
 [4 5 6]]   
 
