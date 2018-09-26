---
layout:     post
title:      Keras入门-使用CNN识别cifar10数据集
keywords:   博客
categories: [机器学习]
tags:	    [Keras]
---

借用Keras官网的介绍：Keras是一个高层神经网络API，Keras由Python编写而成，并能运行在Tensorflow、Theano以及CNTK框架中。Keras 为支持快速实验而生，能够把你的想法迅速转换为结果。如果你有以下需求，请选择Keras：

 - 简易和快速的原型设计
 - 支持CNN和RNN，或两者的结合
 - CPU和GPU无缝切换

鉴于Keras的设计原则：用户友好、模块性、以扩展性和使用python设计，我选择Keras来进行原型设计。


# 使用CNN识别cifar10数据集

cifar10是一个日常物品的数据集，一共有10类，属于是比较小的数据集。以下代码使用一个4个卷积层加2个全连接层的典型CNN网络来进行分类。

	import keras
	from keras.datasets import cifar10
	from keras.preprocessing.image import ImageDataGenerator
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Activation, Flatten
	from keras.layers import Conv2D, MaxPooling2D
	import matplotlib.pyplot as plt
	
	# 载入cifar10 数据集
	(x_train,y_train),(x_test,y_test) = cifar10.load_data()
	
	# 32*32的3通道彩图，训练集5万张，测试集1万张
	print(x_train.shape,y_train.shape)
	print(x_test.shape,y_test.shape)
	# (50000, 32, 32, 3) (50000, 1)
	# (10000, 32, 32, 3) (10000, 1)
	
	plt.imshow(x_train[0])
	plt.show()
	plt.imshow(x_train[1])
	plt.show()
	
	# 数据归一化
	x_train = x_train/255
	x_test = x_test/255
	y_train = keras.utils.to_categorical(y_train,10)
	y_test = keras.utils.to_categorical(y_test,10)

	# 构建模型
	model = Sequential([
	    Conv2D(32,(3,3),padding='same',input_shape=(32,32,3),activation='relu'),
	    Conv2D(32,(3,3),activation='relu'),
	    MaxPooling2D(pool_size=(2,2)),
	    Dropout(0.25),
	    
	    Conv2D(64,(3,3),padding='same',activation='relu'),
	    Conv2D(64,(3,3),activation='relu'),
	    MaxPooling2D(pool_size=(2,2)),
	    Dropout(0.25),
	    
	    Flatten(),
	    Dense(512,activation='relu'),
	    Dropout(0.5),
	    Dense(10,activation='softmax')    
	])

	model.summary()
	
	opt = keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)
	
	model.compile(loss='categorical_crossentropy',
	             optimizer=opt,
	             metrics=['accuracy'])

	# 定义了一个数据增强器，包含了范围20°内的随机旋转，±15%的缩放以及随机的水平翻转
	datagen = ImageDataGenerator(
	    rotation_range = 20,
	    zoom_range = 0.15,
	    horizontal_flip = True,
	)

	# 针对增强数据的训练
	model.fit_generator(datagen.flow(x_train,y_train,batch_size=64),steps_per_epoch = 1000,epochs = 2,validation_data=(x_test,y_test),workers=4,verbose=1)
	
	# 保存模型的结构和参数
	model.save('cifar10_trained_model.h5')
	
	# 评估模型准确率
	scores = model.evaluate(x_test,y_test,verbose=1)
	print('Test loss:',scores[0])
	print('Test accuracy:',scores[1])


# cifar10 数据集载入失败

由于公司网络的限制，cifar10 数据集载入失败。我采用的解决方法是把数据集下载到本地使用。

	(x_train,y_train),(x_test,y_test) = cifar10.load_data()

找到cifar10.py文件（python安装目录\Lib\site-packages\keras\datasets下
）：

	def load_data():
	    """Loads CIFAR10 dataset.
	
	    # Returns
	        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
	    """
	    dirname = 'cifar-10-batches-py'
	    origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
	    path = get_file(dirname, origin=origin, untar=True)

根据origin的链接，可以把数据集下载下来，放到什么地方呢？需要进一步查看data\_utils.py中get_file()函数定义（python安装目录\Lib\site-packages\keras\utils下）

    if untar:
        if not os.path.exists(untar_fpath): 
            _extract_archive(fpath, datadir, archive_format='tar')
        return untar_fpath

我们知道untar为True(见load\_data函数)，我们可以打印出untar\_fpath的值：   

C:\Users\<username>\.keras\datasets\cifar-10-batches-py  

因此我们解压之前下载的数据集（解压后为一个cifar-10-batches-py文件夹），放在C:\Users\<username>\.keras\datasets\目录下即可。还需要做的一件事是修改download变量为False，禁止下载，见下图的206行：   

   ![](/images/images_2018/9-26_01.png)





