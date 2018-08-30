---
layout:     post
title:      幸福之家的人脸识别（编程题）
keywords:   博客
categories: [机器学习]
tags:	    [卷积神经网络，人脸识别]
---

人脸识别问题一般分为两类：一类是人脸验证，例如，在某些机场，您可以通过让系统扫描您的护照来验证您是正确的人来通过海关。使用人脸解锁的手机也使用了人脸验证。这是1：1匹配问题。另一类是人脸识别。例如，视频里显示的百度员工进入办公室的人脸识别系统。这是1：K匹配问题。

# Face Recognition for the Happy House

Welcome to the first assignment of week 4! Here you will build a face recognition system. Many of the ideas presented here are from [FaceNet](https://arxiv.org/pdf/1503.03832.pdf). In lecture, we also talked about [DeepFace](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf). 

Face recognition problems commonly fall into two categories: 

- **Face Verification** - "is this the claimed person?". For example, at some airports, you can pass through customs by letting a system scan your passport and then verifying that you (the person carrying the passport) are the correct person. A mobile phone that unlocks using your face is also using face verification. This is a 1:1 matching problem. 
- **Face Recognition** - "who is this person?". For example, the video lecture showed a face recognition video (https://www.youtube.com/watch?v=wr4rx0Spihs) of Baidu employees entering the office without needing to otherwise identify themselves. This is a 1:K matching problem. 

FaceNet learns a neural network that encodes a face image into a vector of 128 numbers. By comparing two such vectors, you can then determine if two pictures are of the same person.
    
**In this assignment, you will:**
- Implement the triplet loss function
- Use a pretrained model to map face images into 128-dimensional encodings
- Use these encodings to perform face verification and face recognition

In this exercise, we will be using a pre-trained model which represents ConvNet activations using a "channels first" convention, as opposed to the "channels last" convention used in lecture and previous programming assignments. In other words, a batch of images will be of shape $(m, n_C, n_H, n_W)$ instead of $(m, n_H, n_W, n_C)$. Both of these conventions have a reasonable amount of traction among open-source implementations; there isn't a uniform standard yet within the deep learning community. 

Let's load the required packages. 


	from keras.models import Sequential
	from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
	from keras.models import Model
	from keras.layers.normalization import BatchNormalization
	from keras.layers.pooling import MaxPooling2D, AveragePooling2D
	from keras.layers.merge import Concatenate
	from keras.layers.core import Lambda, Flatten, Dense
	from keras.initializers import glorot_uniform
	from keras.engine.topology import Layer
	from keras import backend as K
	K.set_image_data_format('channels_first')
	import cv2
	import os
	import numpy as np
	from numpy import genfromtxt
	import pandas as pd
	import tensorflow as tf
	from fr_utils import *
	from inception_blocks_v2 import *
	
	%matplotlib inline
	%load_ext autoreload
	%autoreload 2
	
	np.set_printoptions(threshold=np.nan)


## 0 - Naive Face Verification

In Face Verification, you're given two images and you have to tell if they are of the same person. The simplest way to do this is to compare the two images pixel-by-pixel. If the distance between the raw images are less than a chosen threshold, it may be the same person! 

   ![](/images/images_2018/pixel_comparison.png)

<center> **Figure 1** </center>

Of course, this algorithm performs really poorly, since the pixel values change dramatically due to variations in lighting, orientation of the person's face, even minor changes in head position, and so on. 

You'll see that rather than using the raw image, you can learn an encoding $f(img)$ so that element-wise comparisons of this encoding gives more accurate judgements as to whether two pictures are of the same person.

要识别两张图片中是不是同一个人，最简单的方法是一个个像素进行对比，当图片的差别小于某个选定的阀值时，就判定为同一个人。但是这种算法表现很差，因为像素值的变化受到多种因素的影响会很大。你将看到将原始图片进行编码后再进行比较，可以更准确地判断两张图片是否是同一个人。

## 1 - Encoding face images into a 128-dimensional vector 

### 1.1 - Using an ConvNet  to compute encodings

The FaceNet model takes a lot of data and a long time to train. So following common practice in applied deep learning settings, let's just load weights that someone else has already trained. The network architecture follows the Inception model from [Szegedy *et al.*](https://arxiv.org/abs/1409.4842). We have provided an inception network implementation. You can look in the file `inception_blocks.py` to see how it is implemented (do so by going to "File->Open..." at the top of the Jupyter notebook). 


- This network uses 96x96 dimensional RGB images as its input. Specifically, inputs a face image (or batch of $m$ face images) as a tensor of shape $(m, n_C, n_H, n_W) = (m, 3, 96, 96)$ 
- It outputs a matrix of shape $(m, 128)$ that encodes each input face image into a 128-dimensional vector

Run the cell below to create the model for face images.


	FRmodel = faceRecoModel(input_shape=(3, 96, 96))
	print("Total Params:", FRmodel.count_params())

执行结果：

Total Params: 3743280


By using a 128-neuron fully connected layer as its last layer, the model ensures that the output is an encoding vector of size 128. You then use the encodings the compare two face images as follows:

   ![](/images/images_2018/distance_kiank.png)

<center> **Figure 2**: By computing a distance between two encodings and thresholding, you can determine if the two pictures represent the same person</center>

So, an encoding is a good one if: 

- The encodings of two images of the same person are quite similar to each other 
- The encodings of two images of different persons are very different

编码将每张图片转换成128维的向量，好的编码需要满足以下两个条件：

- 两张图片中是同一个人时，它们的编码值差异很小；
- 两张图片中不是同一个人时，它们的编码值差异很大；

The triplet loss function formalizes this, and tries to "push" the encodings of two images of the same person (Anchor and Positive) closer together, while "pulling" the encodings of two images of different persons (Anchor, Negative) further apart. 

   ![](/images/images_2018/triplet_comparison.png)

<center> **Figure 3**: In the next part, we will call the pictures from left to right: Anchor (A), Positive (P), Negative (N)  </center>

三元组损失函数，试图推近同一个人的两个图像(Anchor和Positive)的编码，同时，拉开两个不同人物(Anchor和Negative)的图像编码。


### 1.2 - The Triplet Loss

For an image $x$, we denote its encoding $f(x)$, where $f$ is the function computed by the neural network.

   ![](/images/images_2018/f_x.png)

Training will use triplets of images $(A, P, N)$:  

- A is an "Anchor" image--a picture of a person. 
- P is a "Positive" image--a picture of the same person as the Anchor image.
- N is a "Negative" image--a picture of a different person than the Anchor image.

我们用x表示图片，f(x)表示图片的编码函数。训练需要用到图片三元组(A,P,N):

- A 是锚图片，一个人的图片
- P 是正例图片，和锚图片是同一个人
- N 是反例图片，和锚图片是不同的人

   ![](/images/images_2018/8-30_01.png)

Notes:

- The term (1) is the squared distance between the anchor "A" and the positive "P" for a given triplet; you want this to be small. 
- The term (2) is the squared distance between the anchor "A" and the negative "N" for a given triplet, you want this to be relatively large, so it thus makes sense to have a minus sign preceding it. 
- &alpha; is called the margin. It is a hyperparameter that you should pick manually. We will use &alpha; = 0.2. 

Most implementations also normalize the encoding vectors  to have norm equal one (i.e., $\mid \mid f(img)\mid \mid_2$=1); you won't have to worry about that here.

注意：

- 第(1)项是给定三元组的A和P的距离平方，你希望这个值尽可能小
- 第(2)项是给定三元组的A和N的距离平方，你希望这个值尽可能大，因此前
- 面加一个减号是有意义的
- &alpha;是一个超参数，需要你手动选择。这里&alpha; = 0.2


**Exercise**: Implement the triplet loss as defined by formula (3). Here are the 4 steps:

1. Compute the distance between the encodings of "anchor" and "positive"(计算A和P的距离)
2. Compute the distance between the encodings of "anchor" and "negative"(计算A和N的距离)
3. Compute the formula per training example（按照以下公式计算每个训练样本的值）             
   ![](/images/images_2018/8-30_02.jpg)
4. Compute the full formula by taking the max with zero and summing over the training examples.（计算公式J的值）      
   ![](/images/images_2018/8-30_03.jpg)


Useful functions: `tf.reduce_sum()`, `tf.square()`, `tf.subtract()`, `tf.add()`, `tf.maximum()`.
For steps 1 and 2, you will need to sum over the entries of $\mid \mid f(A^{(i)}) - f(P^{(i)}) \mid \mid_2^2$ and $\mid \mid f(A^{(i)}) - f(N^{(i)}) \mid \mid_2^2$ while for step 4 you will need to sum over the training examples.


	# GRADED FUNCTION: triplet_loss
	
	def triplet_loss(y_true, y_pred, alpha = 0.2):
	    """
	    Implementation of the triplet loss as defined by formula (3)
	    
	    Arguments:
	    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
	    y_pred -- python list containing three objects:
	            anchor -- the encodings for the anchor images, of shape (None, 128)
	            positive -- the encodings for the positive images, of shape (None, 128)
	            negative -- the encodings for the negative images, of shape (None, 128)
	    
	    Returns:
	    loss -- real number, value of the loss
	    """
	    
	    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
	    
	    ### START CODE HERE ### (≈ 4 lines)
	    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
	    pos_dist =  tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
	    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
	    neg_dist =  tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
	    # Step 3: subtract the two previous distances and add alpha.
	    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
	    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
	    loss =  tf.reduce_sum(tf.maximum(basic_loss, 0.0))
	    ### END CODE HERE ###
	    
	    return loss

    #test code
	with tf.Session() as test:
	    tf.set_random_seed(1)
	    y_true = (None, None, None)
	    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
	              tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
	              tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
	    loss = triplet_loss(y_true, y_pred)
	    
	    print("loss = " + str(loss.eval()))

执行结果：

loss = 528.143

这里我曾经犯过一个错误：

    loss =  tf.reduce_sum(tf.maximum(basic_loss, 0.0))

我颠倒了tf.reduce_sum()和tf.maximum()两个函数的使用顺序。整体的损失函数应该是每个样本损失值的总和，而每个样本的损失值的计算，需要在0和计算值中取最大值。

## 2 - Loading the trained model

FaceNet is trained by minimizing the triplet loss. But since training requires a lot of data and a lot of computation, we won't train it from scratch here. Instead, we load a previously trained model. Load a model using the following cell; this might take a couple of minutes to run. 


	FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
	load_weights_from_FaceNet(FRmodel)

FaceNet通过最小化三元组损失来进行训练。由于训练需要大量数据和大量计算，因此我们不会从头开始训练，相反，我们加载以前训练过的模型。使用以下单元格加载模型，这可能需要几分钟运行。

Here're some examples of distances between the encodings between three individuals:

   ![](/images/images_2018/distance_matrix.png)

<center>**Figure 4**:Example of distance outputs between three individuals' encodings</center>

Let's now use this model to perform face verification and face recognition! 


注意，这里有一个坑，如果在前面的triplet_loss()函数中，使用了axis = -1，会导致这里模型加载出错:

	    loss =  tf.reduce_sum(tf.maximum(basic_loss, 0.0), axis = -1)

修改如下后可以解决这个错误：

	    loss =  tf.reduce_sum(tf.maximum(basic_loss, 0.0))

在练习中如果遇到一些问题，可以查看论坛的历史信息，相信你遇到的问题，绝大部分能找到答案。


## 3 - Applying the model


Back to the Happy House! Residents are living blissfully since you implemented happiness recognition for the house in an earlier assignment.  

However, several issues keep coming up: The Happy House became so happy that every happy person in the neighborhood is coming to hang out in your living room. It is getting really crowded, which is having a negative impact on the residents of the house. All these random happy people are also eating all your food. 

So, you decide to change the door entry policy, and not just let random happy people enter anymore, even if they are happy! Instead, you'd like to build a **Face verification** system so as to only let people from a specified list come in. To get admitted, each person has to swipe an ID card (identification card) to identify themselves at the door. The face recognition system then checks that they are who they claim to be. 

你决定改变门禁政策，不是仅仅快乐的人就可以进入，相反，你想建立一个“人脸验证”系统，以便只允许指定列表的人进入。每个人必须刷一张身份证，然后由系统检查他们是否是他们声称的人。

### 3.1 - Face Verification

Let's build a database containing one encoding vector for each person allowed to enter the happy house. To generate the encoding we use `img_to_encoding(image_path, model)` which basically runs the forward propagation of the model on the specified image. 

Run the following code to build the database (represented as a python dictionary). This database maps each person's name to a 128-dimensional encoding of their face.

我们为每个允许进入happy house的人建立一个包含编码向量的数据库。为了生成编码，我们使用`img_to_encoding(image_path, model)`，它在指定的图像上运行模型的前向传播。

运行以下代码来构建数据库。该数据库将每个人的姓名映射到他们脸部的128维编码。

	database = {}
	database["danielle"] = img_to_encoding("imagesielle.png", FRmodel)
	database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
	database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
	database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
	database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
	database["dan"] = img_to_encoding("images.jpg", FRmodel)
	database["sebastiano"] = img_to_encoding("imagesbastiano.jpg", FRmodel)
	database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
	database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
	database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
	database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
	database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)


Now, when someone shows up at your front door and swipes their ID card (thus giving you their name), you can look up their encoding in the database, and use it to check if the person standing at the front door matches the name on the ID.

**Exercise**: Implement the verify() function which checks if the front-door camera picture (`image_path`) is actually the person called "identity". You will have to go through the following steps:
1. Compute the encoding of the image from image_path
2. Compute the distance about this encoding and the encoding of the identity image stored in the database
3. Open the door if the distance is less than 0.7, else do not open.

As presented above, you should use the L2 distance (np.linalg.norm). (Note: In this implementation, compare the L2 distance, not the square of the L2 distance, to the threshold 0.7.) 

现在，当有人刷身份证想进入happy house时（获得他们的名字），你可以在数据库中查找他们的编码，并用它来检查站在前门的人是否是与名字相符的那个人。

练习需要实现verify()函数。执行步骤如下：

1. 从image_path计算图像的编码
2. 计算该编码和存储在数据库中的身份图像的编码的距离
3. 如果距离小于0.7，请打开门，否则不打开

-----------------------------------------

	# GRADED FUNCTION: verify
	
	def verify(image_path, identity, database, model):
	    """
	    Function that verifies if the person on the "image_path" image is "identity".
	    
	    Arguments:
	    image_path -- path to an image
	    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
	    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
	    model -- your Inception model instance in Keras
	    
	    Returns:
	    dist -- distance between the image_path and the image of "identity" in the database.
	    door_open -- True, if the door should open. False otherwise.
	    """
	    
	    ### START CODE HERE ###
	    
	    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
	    encoding = img_to_encoding(image_path, model)
	    
	    # Step 2: Compute distance with identity's image (≈ 1 line)
	    dist =  np.linalg.norm(encoding-database[identity])
	    
	    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
	    if dist < 0.7 :
	        print("It's " + str(identity) + ", welcome home!")
	        door_open = True
	    else:
	        print("It's not " + str(identity) + ", please go away")
	        door_open = False
	        
	    ### END CODE HERE ###
	        
	    return dist, door_open

    #test code
    verify("images/camera_0.jpg", "younes", database, FRmodel)

执行结果：

It's younes, welcome home!
(0.65939283, True)


Benoit, who broke the aquarium last weekend, has been banned from the house and removed from the database. He stole Kian's ID card and came back to the house to try to present himself as Kian. The front-door camera took a picture of Benoit ("images/camera_2.jpg). Let's run the verification algorithm to check if benoit can enter.

Benoit不在数据库中，他偷了Kian的身份证尝试进入happy house，让我们看看他是否能进入。


    verify("images/camera_2.jpg", "kian", database, FRmodel)

执行结果：

It's not kian, please go away
(0.86224014, False)

### 3.2 - Face Recognition

Your face verification system is mostly working well. But since Kian got his ID card stolen, when he came back to the house that evening he couldn't get in! 

To reduce such shenanigans, you'd like to change your face verification system to a face recognition system. This way, no one has to carry an ID card anymore. An authorized person can just walk up to the house, and the front door will unlock for them! 

You'll implement a face recognition system that takes as input an image, and figures out if it is one of the authorized persons (and if so, who). Unlike the previous face verification system, we will no longer get a person's name as another input. 

你的人脸验证系统运作良好，但是如果你的身份证被偷了，那么就无法进入happy house了。为了避免这种情况，你需要将人脸验证系统更改为人脸识别系统。这样，就不用携带身份证了。获得授权的人可以走到房子前面，前门会为他们解锁。

你将实现一个人脸识别系统，该系统将图像作为输入，并确定它是否是授权人员。与之前的人脸验证系统不同，我们将不再将人名作为另一个输入。

**Exercise**: Implement `who_is_it()`. You will have to go through the following steps:

1. Compute the target encoding of the image from image_path
2. Find the encoding from the database that has smallest distance with the target encoding. 
    - Initialize the `min_dist` variable to a large enough number (100). It will help you keep track of what is the closest encoding to the input's encoding.
    - Loop over the database dictionary's names and encodings. To loop use `for (name, db_enc) in database.items()`.
        - Compute L2 distance between the target "encoding" and the current "encoding" from the database.
        - If this distance is less than the min_dist, then set min_dist to dist, and identity to name.

练习：实现`who_is_it()`，必须执行以下步骤：

1. 从image_path 计算图像的目标编码
2. 从数据库中找到与目标编码具有最小距离的编码
   - `min_dist`初始化为足够大的数值(100)。它将帮助你跟踪输入编码的最接近编码
   - 遍历数据库字典的名称和编码。循环使用 `for (name, db_enc) in database.items()`
     - 计算目标编码与数据库中当前编码之间的L2距离
     - 如果此距离小于`min_dist`，则将`min_dist`设置为dist，将identity设置为name


----------------------------

	# GRADED FUNCTION: who_is_it
	
	def who_is_it(image_path, database, model):
	    """
	    Implements face recognition for the happy house by finding who is the person on the image_path image.
	    
	    Arguments:
	    image_path -- path to an image
	    database -- database containing image encodings along with the name of the person on the image
	    model -- your Inception model instance in Keras
	    
	    Returns:
	    min_dist -- the minimum distance between image_path encoding and the encodings from the database
	    identity -- string, the name prediction for the person on image_path
	    """
	    
	    ### START CODE HERE ### 
	    
	    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
	    encoding = img_to_encoding(image_path, model)
	    
	    ## Step 2: Find the closest encoding ##
	    
	    # Initialize "min_dist" to a large value, say 100 (≈1 line)
	    min_dist = 100
	    
	    # Loop over the database dictionary's names and encodings.
	    for (name, db_enc) in database.items():
	        
	        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
	        dist = np.linalg.norm(encoding - db_enc)
	
	        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
	        if dist < min_dist :
	            min_dist = dist
	            identity = name
	
	    ### END CODE HERE ###
	    
	    if min_dist > 0.7:
	        print("Not in the database.")
	    else:
	        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
	        
	    return min_dist, identity

Younes is at the front-door and the camera takes a picture of him ("images/camera_0.jpg"). Let's see if your who_it_is() algorithm identifies Younes. 

    who_is_it("images/camera_0.jpg", database, FRmodel)

执行结果：

it's younes, the distance is 0.659393
(0.65939283, 'younes')

You can change "`camera_0.jpg`" (picture of younes) to "`camera_1.jpg`" (picture of bertrand) and see the result.


Your Happy House is running well. It only lets in authorized persons, and people don't need to carry an ID card around anymore! 

You've now seen how a state-of-the-art face recognition system works.

Although we won't implement it here, here're some ways to further improve the algorithm:

- Put more images of each person (under different lighting conditions, taken on different days, etc.) into the database. Then given a new image, compare the new face to multiple pictures of the person. This would increae accuracy.
- Crop the images to just contain the face, and less of the "border" region around the face. This preprocessing removes some of the irrelevant pixels around the face, and also makes the algorithm more robust.

你的happy house运行良好。它只允许授权人员，人们不再需要携带身份证了。你现在已经了解了最先进的人脸识别系统的工作原理。虽然我们不会在这里实现它，但是有一些方法可以进一步改进算法：

- 将每个人的更多图像(在不同的光照条件下，在不同的日子拍摄等)放入数据库。然后给出一个新图像，将新人脸与该人的多个图像进行比较。这将提高准确性。
- 裁剪图像仅包含人脸，并减少人脸周围的边框区域。这种预处理消除了人脸周围的一些无关像素，并使算法更加健壮。


**What you should remember**:

- Face verification solves an easier 1:1 matching problem; face recognition addresses a harder 1:K matching problem. 
- The triplet loss is an effective loss function for training a neural network to learn an encoding of a face image.
- The same encoding can be used for verification and recognition. Measuring distances between two images' encodings allows you to determine whether they are pictures of the same person.


**划重点** ：

- 人脸验证解决的是更简单的1：1匹配问题；人脸识别解决的是更难的1：K匹配问题
- 三元组损失是用于训练神经网络以学习人脸图像的编码的有效损失函数
- 相同的编码可用于验证和识别。测量两个图像编码之间的距离可以确定它们是否是同一个人的图像


### References:

- Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
- Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) 
- The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
- Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet 
