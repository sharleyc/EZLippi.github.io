---
layout:     post
title:      Keras入门-数据集载入源码分析
keywords:   博客
categories: [机器学习]
tags:	    [Keras]
---

刚开始学习Keras，大家一般都是基于MNIST手写数据集、CIFAR系列数据集来实践的。这些数据集的特点是已经为用户打包封装好了数据，在Keras环境下只要load_data就可以实现导入。在实际项目中，我们往往需要自己建议一个数据集。本文通过对keras源码中数据集载入的分析，来尝试解决这个问题。


# cifar10数据集载入

载入cifar10数据集只需要一句代码：

	(x_train,y_train),(x_test,y_test) = cifar10.load_data()
	
我们先看看load_data()的源码，简单的说了做了两件事：一个是下载数据集(如果本地没有数据集的话)，一个是处理数据集为keras模型要求的格式。


	"""CIFAR10 small images classification dataset.
	"""
	from __future__ import absolute_import
	from __future__ import division
	from __future__ import print_function
	
	from .cifar import load_batch
	from ..utils.data_utils import get_file
	from .. import backend as K
	import numpy as np
	import os
	
	
	def load_data():
	    """Loads CIFAR10 dataset.
	
	    # Returns
	        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
	    """
	    dirname = 'cifar-10-batches-py'
	    origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
	    path = get_file(dirname, origin=origin, untar=True)
	
	    num_train_samples = 50000
	
	    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
	    y_train = np.empty((num_train_samples,), dtype='uint8')
	
	    for i in range(1, 6):
	        fpath = os.path.join(path, 'data_batch_' + str(i))
	        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
	         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)
	
	    fpath = os.path.join(path, 'test_batch')
	    x_test, y_test = load_batch(fpath)
	
	    y_train = np.reshape(y_train, (len(y_train), 1))
	    y_test = np.reshape(y_test, (len(y_test), 1))
	
	    if K.image_data_format() == 'channels_last':
	        x_train = x_train.transpose(0, 2, 3, 1)
	        x_test = x_test.transpose(0, 2, 3, 1)
	
	    return (x_train, y_train), (x_test, y_test)

我们看看数据集的结构：训练集的数据分成了5个data_batch，测试集的数据是test_batch，并且每个batch是10000条数据，数据集的载入工作实际是由load_batch()完成的。

   ![](/images/images_2018/9-28_01.png)

我们接着看看load_batch()的源码：
(keras/keras/datasets/cifar.py)

	# -*- coding: utf-8 -*-
	"""Utilities common to CIFAR10 and CIFAR100 datasets.
	"""
	from __future__ import absolute_import
	from __future__ import division
	from __future__ import print_function
	
	import sys
	from six.moves import cPickle
	
	
	def load_batch(fpath, label_key='labels'):
	    """Internal utility for parsing CIFAR data.
	    # Arguments
	        fpath: path the file to parse.
	        label_key: key for label data in the retrieve
	            dictionary.
	    # Returns
	        A tuple `(data, labels)`.
	    """
	    with open(fpath, 'rb') as f:
	        if sys.version_info < (3,):
	            d = cPickle.load(f)
	        else:
	            d = cPickle.load(f, encoding='bytes')
	            # decode utf8
	            d_decoded = {}
	            for k, v in d.items():
	                d_decoded[k.decode('utf8')] = v
	            d = d_decoded
	    data = d['data']
	    labels = d[label_key]
	
	    data = data.reshape(data.shape[0], 3, 32, 32)
	    return data, labels

如果是Python2.*版本，则调用cPickle.load(f)函数，如果是Python3.*版本，则调用cPickle.load(f, encoding='byte')。显而易见，关键是cPickle.load()函数。在Python3里，cPickle库已经被废弃，替代使用的是pickle库。由于Python3是大势所趋，这里仅看pickle库。


# pickle库 了解一下

pickle库实现了基本的数据序列化和反序列化。序列化过程是将文本信息转变为二进制数据流，便于存储和传输。反序列化就是把二进制数据流转变为文本信息。pickle_load()实现的是将序列化的对象从文件中读取出来，且必须以二进制的形式('rb')进行读取，返回的是一个字典。

	def unpickle(file):
	    import pickle
	    with open(file, 'rb') as fo:
	        dict = pickle.load(fo, encoding='bytes')
	    return dict

pickle_dump()则相反，实现的是将文本信息转变为二进制数据流。由此不难得到我们可以利用pickle_dump()来达到存储我们的数据集的目的。（以下是help显示的部分信息）

	dump(obj, file, protocol=None, *, fix_imports=True)
	    Write a pickled representation of obj to the open file object file.
	
	    This is equivalent to ``Pickler(file, protocol).dump(obj)``, but may
	    be more efficient.

	load(file, *, fix_imports=True, encoding='ASCII', errors='strict')
	    Read and return an object from the pickle data stored in a file.
	
	    This is equivalent to ``Unpickler(file).load()``, but may be more
	    efficient.

# 使用pickle进行图片序列化以及读取显示


 






