---
layout:     post
title:      机器学习策略-自动驾驶（测试题）
keywords:   博客
categories: [机器学习]
tags:	    [机器学习，ML Strategy]
---

为了帮助你练习机器学习策略，本周将介绍另一种情况，并询问你将如何行动。我们认为这个模拟器将会帮助你了解到一个机器学习项目是如何进行的。


你受雇于一家研究自动驾驶的创业公司。你负责检测图像中的道路标志（停车标志，人行横道标志，施工前方标志）和交通信号灯（红灯和绿灯）。 目标是识别每个图像中出现哪些对象。例如，下面的图像包含行人过路标志和红色交通灯

   ![](/images/images_2018/8-1_01.png) 

你的100,000张标签图像是使用汽车的前置摄像头拍摄的。 这也是你最关心的数据分布。 你认为你可能能够从互联网上获得更大的数据集，即使互联网数据的分布不同，这对训练也很有帮助。


## 第1题

You are just getting started on this project. What is the first thing you do? Assume each of the steps below would take about an equal amount of time (a few days).

- Spend a few days collecting more data using the front-facing camera of your car, to better understand how much data per unit time you can collect.
- Spend a few days getting the internet data, so that you understand better what data is available.
- Spend a few days checking what is human-level performance for these tasks so that you can get an accurate estimate of Bayes error.
- Spend a few days training a basic model and see what mistakes it makes.

正确答案： 第4项。ML是一个快速迭代的过程，当你建立一个基本的模型后，就可以进行误差分析，了解下一步工作的方向。

-------------------------------------

## 第2题

Your goal is to detect road signs (stop sign, pedestrian crossing sign, construction ahead sign) and traffic signals (red and green lights) in images. The goal is to recognize which of these objects appear in each image. You plan to use a deep neural network with ReLU units in the hidden layers.

For the output layer, a softmax activation would be a good choice for the output layer because this is a multi-task learning problem. True/False?    

正确答案：False。softmax函数适用于有且仅有一个项的取值为1。而多任务学习中，存在多个项的取值为1的可能。

---------------------------------------------

## 第3题  

You are carrying out error analysis and counting up what errors the algorithm makes. Which of these datasets do you think you should manually go through and carefully examine, one image at a time?


- 500 images on which the algorithm made a mistake
- 10,000 images on which the algorithm made a mistake
- 10,000 randomly chosen images
- 500 randomly chosen images

正确答案：第1个选项。算法分析错误的500张图片。

---------------------------------------

## 第4题 

After working on the data for several weeks, your team ends up with the following data:

- 100,000 labeled images taken using the front-facing camera of your car.
- 900,000 labeled images of roads downloaded from the internet.
- Each image’s labels precisely indicate the presence of any specific road signs and traffic signals or combinations of them. Because this is a multi-task learning problem, you need to have all your y^{(i)}y 
(i) vectors fully labeled. If some entries of the example haven't been labeled(？), then the learning algorithm will not be able to use that example. True/False?

正确答案： False。某些样本没有标签，并不影响计算代价函数。

---------------------------------------

## 第5题

The distribution of data you care about contains images from your car’s front-facing camera; which comes from a different distribution than the images you were able to find and download off the internet. How should you split the dataset into train/dev/test sets?

- Mix all the 100,000 images with the 900,000 images you found online. Shuffle everything. Split the 1,000,000 images dataset into 980,000 for the training set, 10,000 for the dev set and 10,000 for the test set.
- Choose the training set to be the 900,000 images from the internet along with 20,000 images from your car’s front-facing camera. The 80,000 remaining images will be split equally in dev and test sets.
- Mix all the 100,000 images with the 900,000 images you found online. Shuffle everything. Split the 1,000,000 images dataset into 600,000 for the training set, 200,000 for the dev set and 200,000 for the test set.
- Choose the training set to be the 900,000 images from the internet along with 80,000 images from your car’s front-facing camera. The 20,000 remaining images will be split equally in dev and test sets.

正确答案： 第4个选项。保证开发集和测试集具有相同的分布，且符合目标数据的特征是很重要的，第2、4个选项是错误的；在保证dev/test集有足够数据的前提下，一般来说，训练集的数据越大效果越好，因此选择第4个选项。

----------------------------------------------

## 第6题

Assume you’ve finally chosen the following split between of the data:

- Training	940,000 images randomly picked from (900,000 internet images + 60,000 car’s front-facing camera images)	8.8%
- Training-Dev	20,000 images randomly picked from (900,000 internet images + 60,000 car’s front-facing camera images)	9.1%
- Dev	20,000 images from your car’s front-facing camera	14.3%
- Test	20,000 images from the car’s front-facing camera	14.8%

You also know that human-level error on the road sign and traffic signals classification task is around 0.5%. Which of the following are True? (Check all that apply).

- You have a large avoidable-bias problem because your training error is quite a bit higher than the human-level error.
- You have a large data-mismatch problem because your model does a lot better on the training-dev set than on the dev set
- You have a large variance problem because your training error is quite higher than the human-level error.
- You have a large variance problem because your model is not generalizing well to data from the same training distribution but that it has never seen before.
- Your algorithm overfits the dev set because the error of the dev and test sets are very close.


正确答案： 第1、2个选项。人类水平误差是0.5%，训练集误差是8.8%，训练集-开发集误差是9.1%，开发集误差14.3%，测试集误差14.8%。因为训练集误差远大于人类水平误差，因此肯定是有很大的avoidable-bias error；而训练集误差和训练集-开发集误差是比较接近的，说明并没有很大的variance error；而训练集误差和开发集误差比较大，说明有比较大的data-mismatch problem。因此应该选择第1、2个选项。

------------------------------------------

## 第7题

Based on table from the previous question, a friend thinks that the training data distribution is much easier than the dev/test distribution. What do you think?

- Your friend is right. (I.e., Bayes error for the training data distribution is probably lower than for the dev/test distribution.)
- Your friend is wrong. (I.e., Bayes error for the training data distribution is probably higher than for the dev/test distribution.)
- There’s insufficient information to tell if your friend is right or wrong.

正确答案：第3项。原因有可能是训练集的难度远低于实际数据集，也有可能是模型有较大偏差，这里并没有给出足够的信息下定论。

----------------------------------------

## 第8题

You decide to focus on the dev set and check by hand what are the errors due to. Here is a table summarizing your discoveries:

Overall dev set error	14.3%   
Errors due to incorrectly labeled data	4.1%   
Errors due to foggy pictures	8.0%    
Errors due to rain drops stuck on your car’s front-facing camera	2.2%    
Errors due to other causes	1.0%    

In this table, 4.1%, 8.0%, etc.are a fraction of the total dev set (not just examples your algorithm mislabeled). I.e. about 8.0/14.3 = 56% of your errors are due to foggy pictures.

The results from this analysis implies that the team’s highest priority should be to bring more foggy pictures into the training set so as to address the 8.0% of errors in that category. True/False?

- True because it is the largest category of errors. As discussed in lecture, we should prioritize the largest category of error to avoid wasting the team’s time.
- True because it is greater than the other error categories added together (8.0 > 4.1+2.2+1.0).
- False because this would depend on how easy it is to add this data and how much you think your team thinks it’ll help.
- False because data augmentation (synthesizing foggy images by clean/non-foggy images) is more efficient.

正确答案：第3项。猜测。

-----------------------------

## 第9题

You can buy a specially designed windshield wiper that help wipe off some of the raindrops on the front-facing camera. Based on the table from the previous question, which of the following statements do you agree with?

- 2.2% would be a reasonable estimate of the maximum amount this windshield wiper could improve performance.
- 2.2% would be a reasonable estimate of the minimum amount this windshield wiper could improve performance.
- 2.2% would be a reasonable estimate of how much this windshield wiper will improve performance.
- 2.2% would be a reasonable estimate of how much this windshield wiper could worsen performance in the worst case.

正确答案：第1项。2.2%是表现上限，即最好的情况下能提升2.2%。

--------------------------------------

## 第10题

You decide to use data augmentation to address foggy images. You find 1,000 pictures of fog off the internet, and “add” them to clean images to synthesize foggy days, like this:


Which of the following statements do you agree with?

- So long as the synthesized fog looks realistic to the human eye, you can be confident that the synthesized data is accurately capturing the distribution of real foggy images (or a subset of it), since human vision is very accurate for the problem you’re solving.
- There is little risk of overfitting to the 1,000 pictures of fog so long as you are combing it with a much larger (>>1,000) of clean/non-foggy images.
- Adding synthesized images that look like real foggy pictures taken from the front-facing camera of your car to training dataset won’t help the model improve because it will introduce avoidable-bias.

正确答案：第1项。只要合成的图片看起来足够真实，能帮助模型提高在有雾的天气下识别道路信号的能力。

----------------------------------------

## 第11题

After working further on the problem, you’ve decided to correct the incorrectly labeled data on the dev set. Which of these statements do you agree with? (Check all that apply).

- You should also correct the incorrectly labeled data in the test set, so that the dev and test sets continue to come from the same distribution
- You should correct incorrectly labeled data in the training set as well so as to avoid your training set now being even more different from your dev set.
- You should not correct the incorrectly labeled data in the test set, so that the dev and test sets continue to come from the same distribution
- You should not correct incorrectly labeled data in the training set as it does not worth the time.

正确答案：第1、4项。为了确保开发和测试集数据具有相同的分布，因此要选择第1项；深度学习算法对于随机错误非常健壮，对于占比不高的训练集中的随机错误不值得花时间去纠正，因此选择第4项。

---------------------------------------

## 第12题

So far your algorithm only recognizes red and green traffic lights. One of your colleagues in the startup is starting to work on recognizing a yellow traffic light. (Some countries call it an orange light rather than a yellow light; we’ll use the US convention of calling it yellow.) Images containing yellow lights are quite rare, and she doesn’t have enough data to build a good model. She hopes you can help her out using transfer learning.

What do you tell your colleague?

- She should try using weights pre-trained on your dataset, and fine-tuning further with the yellow-light dataset.
- If she has (say) 10,000 images of yellow lights, randomly sample 10,000 images from your dataset and put your and her data together. This prevents your dataset from “swamping” the yellow lights dataset.
- You cannot help her because the distribution of data you have is different from hers, and is also lacking the yellow label.
- Recommend that she try multi-task learning instead of transfer learning using all the data.

正确答案：第1项。这是迁移学习的完美案例。你已经在庞大的数据集上训练了模型，并且你的同事有一个小数据集，虽然标签不同，但是模型参数经过训练，可以识别道路和交通图像的许多特征，而这些特征对她的问题很有用。她可以从和你相同架构的模型开始，更改最后一个隐藏层之后的内容，并使用你训练的参数对其进行初始化。

---------------------------------------

## 第13题

Another colleague wants to use microphones placed outside the car to better hear if there’re other vehicles around you. For example, if there is a police vehicle behind you, you would be able to hear their siren. However, they don’t have much to train this audio system. How can you help?


- Transfer learning from your vision dataset could help your colleague get going faster. Multi-task learning seems significantly less promising.
- Multi-task learning from your vision dataset could help your colleague get going faster. Transfer learning seems significantly less promising.
- Either transfer learning or multi-task learning could help our colleague get going faster.
- Neither transfer learning nor multi-task learning seems promising.

正确答案： 第4项。一个是图像识别问题，一个是语音识别问题，两个问题完全不同，需要的数据集类型也完全不同，因此既无法使用迁移学习，也无法使用多任务学习。

----------------------------------------------

## 第14题

To recognize red and green lights, you have been using this approach:

(A) Input an image (x) to a neural network and have it directly learn a mapping to make a prediction as to whether there’s a red light and/or green light (y).
A teammate proposes a different, two-step approach:

(B) In this two-step approach, you would first (i) detect the traffic light in the image (if any), then (ii) determine the color of the illuminated lamp in the traffic light.
Between these two, Approach B is more of an end-to-end approach because it has distinct steps for the input end and the output end. True/False?

正确答案：False。(A) 方法是端到端的方法，因为它直接从输入(x) 映射到输出(y)。

---------------------------------------

## 第15题

Approach A (in the question above) tends to be more promising than approach B if you have a ________ (fill in the blank).

- Large training set
- Multi-task learning problem.
- Large bias problem.
- Problem with a high Bayes error.


正确答案：第1项。在许多领域，已经观察到端到端的深度学习在实践中能更好的工作，但是需要大量数据。



	
