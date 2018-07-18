---
layout:     post
title:      np.dot和*的区别    
keywords:   博客
categories: [Python]
tags:	    [numpy, dot]
---

## array()和matrix()在矩阵乘法中的不同表现

	import numpy as np 
	
	a = np.array([[1,2],[3,4]])
	b = np.array([[5,6],[7,8]])
	
	c = np.mat([[1,2],[3,4]])
	d = np.mat([[5,6],[7,8]])
	
	print(a * b)
	print(c * d)
    print(np.dot(a, b))

执行结果：  

	[[ 5 12]  
	 [21 32]]  
	[[19 22]  
	 [43 50]]
	[[19 22]  
	 [43 50]]

以上输出结果会有不同，原因是array()函数的乘法(*)是矩阵对应位置的两个数进行相乘，而mat()函数总是遵循矩阵乘法规则。 如果需要array()函数遵循矩阵乘法规则，则需要使用np.dot()函数。 

## 关于np.dot()和*的区别   

	A = np.arange(1,4)
	B = np.arange(0,3)
	print(np.dot(A,B))
	C = np.arange(1,5).reshape(2,2)
	D = np.arange(0,4).reshape(2,2)
	print(np.dot(C,D))
	
执行结果： 

	8
	[[ 4  7]
	 [ 8 15]]

A = array([1,2,3]), B = array([0,1,2]), A和B是秩为1的数组（即一维数组），对于秩为1的数组，执行对应位置相乘，然后再相加；C和D是二维数组，对于秩不为1的数组，执行矩阵乘法运算。   



参考资料：  

https://blog.csdn.net/lfj742346066/article/details/77880668    
https://blog.csdn.net/zenghaitao0128/article/details/78715140  
