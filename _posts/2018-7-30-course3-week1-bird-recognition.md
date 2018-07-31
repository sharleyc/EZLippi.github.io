---
layout:     post
title:      机器学习策略-鸟类识别（测试题）
keywords:   博客
categories: [机器学习]
tags:	    [机器学习，ML Strategy]
---

# 问题陈述

此示例改编自真实的生产应用程序，里面有一些虚构的细节是为了保护机密性。

   ![](/images/images_2018/7-30_01.jpg) 

你是Peacetopia市著名的研究员。Peacetopia的人有一个共同的特点：他们害怕鸟类。为了保存他们，你必须建立一个算法，以检测任何飞越Peaceopia的鸟并提醒人们。

市议会为你提供了10,000,000张Peacetopia上空天空图像的数据集，这些图像取自该市的安全摄像头。 它们被标记为：   

- y = 0：图像上没有鸟   
- y = 1：图像上有鸟

你的目标是构建一种能够对Peacetopia安全摄像头拍摄的新图像进行分类的算法。

你需要做出一系列的决定：

- 什么是评估指标？    
- 如何将数据构建到train/dev/test集中？



## 第1题

Metric of success

The City Council tells you that they want an algorithm that

- Has high accuracy
- Runs quickly and takes only a short time to classify a new image.
- Can fit in a small amount of memory, so that it can run in a small processor that the city will attach to many different security cameras.   

Note: Having three evaluation metrics makes it harder for you to quickly choose between two different algorithms, and will slow down the speed with which your team can iterate. True/False?

正确答案： True。单一数值的衡量标准会有利于快速判断不同算法的优劣。

-------------------------------------

## 第2题

After further discussions, the city narrows down its criteria to:

- "We need an algorithm that can let us know a bird is flying over Peacetopia as accurately as possible."
- "We want the trained model to take no more than 10sec to classify a new image.”
- “We want the model to fit in 10MB of memory.”

If you had the three following models, which one would you choose?

-  Test Accuracy	97% Runtime 1 sec	Memory size 3MB    
-  Test Accuracy	99% Runtime 13 sec	Memory size 9MB     
-  Test Accuracy	97% Runtime 3 sec	Memory size 2MB    
-  Test Accuracy	98% Runtime 9 sec	Memory size 9MB    

正确答案：第四个选项。Test Accuracy是优化指标，Runtime和Memory size是满足指标。B的满足指标不符合要求，被淘汰。剩下的ACD的满足指标均符合要求，因此关键看优化指标，D的Accuracy是最高的，因此选择D。

---------------------------------------------

## 第3题  

Based on the city’s requests, which of the following would you say is true?

- Accuracy is an optimizing metric; running time and memory size are a satisficing metrics.
- Accuracy is a satisficing metric; running time and memory size are an optimizing metric.
- Accuracy, running time and memory size are all optimizing metrics because you want to do well on all three.
- Accuracy, running time and memory size are all satisficing metrics because you have to do sufficiently well on all three for your system to be acceptable.

答案：第一个选项。

---------------------------------------

## 第4题 

Structuring your data

Before implementing your algorithm, you need to split your data into train/dev/test sets. Which of these do you think is the best choice?

- Train	Dev	Test  6,000,000	3,000,000	1,000,000
- Train	Dev	Test  3,333,334	3,333,333	3,333,333
- Train	Dev	Test  9,500,000	250,000	250,000
- Train	Dev	Test  6,000,000	1,000,000	3,000,000

答案： 第三个选项。考虑到样本数量10,000,000，dev & test的数量足够用即可，大部分应该用于train set。

---------------------------------------

## 第5题

After setting up your train/dev/test sets, the City Council comes across another 1,000,000 images, called the “citizens’ data”. Apparently the citizens of Peacetopia are so scared of birds that they volunteered to take pictures of the sky and label them, thus contributing these additional 1,000,000 images. These images are different from the distribution of images the City Council had originally given you, but you think it could help your algorithm.

You should not add the citizens’ data to the training set, because this will cause the training and dev/test set distributions to become different, thus hurting dev and test set performance. True/False?

正确答案： False。将新的数据集加入到训练集不会影响dev/test集的分布，因此上述说法是错误的。

----------------------------------------------

## 第6题

One member of the City Council knows a little about machine learning, and thinks you should add the 1,000,000 citizens’ data images to the test set. You object because:

- The test set no longer reflects the distribution of data (security cameras) you most care about.
- A bigger test set will slow down the speed of iterating because of the computational expense of evaluating models on the test set.
- This would cause the dev and test set distributions to become different. This is a bad idea because you’re not aiming where you want to hit.
- The 1,000,000 citizens’ data images do not have a consistent x-->y mapping as the rest of the data (similar to the New York City/Detroit housing prices example from lecture).

正确答案： 第1、3选项。新增的图像数据不符合之前设定的目标（不是来自安全摄像头的数据），会导致开发测试集的数据分布不一致。

------------------------------------------

## 第7题

You train a system, and its errors are as follows (error = 100%-Accuracy):

Training set error	4.0%
Dev set error	4.5%

This suggests that one good avenue for improving performance is to train a bigger network so as to drive down the 4.0% training error. Do you agree?


- Yes, because having 4.0% training error shows you have high bias.
- Yes, because this shows your bias is higher than your variance.
- No, because this shows your variance is higher than your bias.
- No, because there is insufficient information to tell.

正确答案：第4项。缺少human-level error，无法判断bias error & variance error哪个更高。

----------------------------------------

## 第8题

You ask a few people to label the dataset so as to find out what is human-level performance. You find the following levels of accuracy:

- Bird watching expert #1	0.3% error
- Bird watching expert #2	0.5% error
- Normal person #1 (not a bird watching expert)	1.0% error
- Normal person #2 (not a bird watching expert)	1.2% error

You ask a few people to label the dataset so as to find out what is human-level performance. You find the following levels of accuracy:

If your goal is to have “human-level performance” be a proxy (or estimate) for Bayes error, how would you define “human-level performance”?

- 0.0% (because it is impossible to do better than this)
- 0.3% (accuracy of expert #1)
- 0.4% (average of 0.3 and 0.5)
- 0.75% (average of all four numbers above)

正确答案：第2项。考虑这里的语境是评估Bayes error，应该选择最低的人类错误率。

-----------------------------

## 第9题

Which of the following statements do you agree with?

- A learning algorithm’s performance can be better than human-level performance but it can never be better than Bayes error.
- A learning algorithm’s performance can never be better than human-level performance but it can be better than Bayes error.
- A learning algorithm’s performance can never be better than human-level performance nor better than Bayes error.
- A learning algorithm’s performance can be better than human-level performance and better than Bayes error.

正确答案：第1项。学习算法的正确率是可以比人类的更高的。但是不会超过Bayes error。

--------------------------------------

## 第10题

You find that a team of ornithologists debating and discussing an image gets an even better 0.1% performance, so you define that as “human-level performance.” After working further on your algorithm, you end up with the following:

Human-level performance	0.1%
Training set error	2.0%
Dev set error	2.1%

Based on the evidence you have, which two of the following four options seem the most promising to try? (Check two options.)

- Try decreasing regularization.
- Train a bigger model to try to do better on the training set.
- Try increasing regularization.
- Get a bigger training set to reduce variance.

正确答案：第1、2项。1.9% > 0.1%，should focus on bias。

----------------------------------------

## 第11题

You also evaluate your model on the test set, and find the following:

Human-level performance	0.1%
Training set error	2.0%
Dev set error	2.1%
Test set error	7.0%

What does this mean? (Check the two best options.)

- You should get a bigger test set.
- You have overfit to the dev set.
- You should try to get a bigger dev set.
- You have underfit to the dev set.

正确答案：第2、3项。从数据看模型的泛化能力比较差，意味过拟合了，需要更多的数据集。

---------------------------------------

## 第12题

After working on this project for a year, you finally achieve:

Human-level performance	0.10%
Training set error	0.05%
Dev set error	0.05%
What can you conclude? (Check all that apply.)

- If the test set is big enough for the 0.05% error estimate to be accurate, this implies Bayes error is ≤0.05
- It is now harder to measure avoidable bias, thus progress will be slower going forward.
- With only 0.09% further progress to make, you should quickly be able to close the remaining gap to 0%
- This is a statistical anomaly (or must be the result of statistical noise) since it should not be possible to surpass human-level performance.

正确答案：第1、2项。算法的准确率高于人类水平的准确率后，因为缺少改善算法的重要手段，其进展会大大减慢。而算法的准确度是不可能超过Bayes error的，当测试集足够大时，我们可以认为Bayes error是<=0.05%的。

---------------------------------------

## 第13题

It turns out Peacetopia has hired one of your competitors to build a system as well. Your system and your competitor both deliver systems with about the same running time and memory size. However, your system has higher accuracy! However, when Peacetopia tries out your and your competitor’s systems, they conclude they actually like your competitor’s system better, because even though you have higher overall accuracy, you have more false negatives (failing to raise an alarm when a bird is in the air). What should you do?

- Look at all the models you’ve developed during the development process and find the one with the lowest false negative error rate.
- Ask your team to take into account both accuracy and false negative rate during development.
- Rethink the appropriate metric for this task, and ask your team to tune to the new metric.
- Pick false negative rate as the new metric, and use this new metric to drive all further development.

正确答案： 第3项。当不满意原来的衡量指标时，应尝试新指标。

----------------------------------------------

## 第14题

You’ve handily beaten your competitor, and your system is now deployed in Peacetopia and is protecting the citizens from birds! But over the last few months, a new species of bird has been slowly migrating into the area, so the performance of your system slowly degrades because your data is being tested on a new type of data.

You have only 1,000 images of the new species of bird. The city expects a better system from you within the next 3 months. Which of these should you do first?

- Use the data you have to define a new evaluation metric (using a new dev/test set) taking into account the new species, and use that to drive further progress for your team.
- Put the 1,000 images into the training set so as to try to do better on these birds.
- Try data augmentation/data synthesis to get more images of the new type of bird.
- Add the 1,000 images into your dataset and reshuffle into a new train/dev/test split.

正确答案：第1项。预测目标发生了变化，首先需要重新确认目标，意味着要定义新的衡量指标，然后是精确瞄准。

---------------------------------------

## 第15题

The City Council thinks that having more Cats in the city would help scare off birds. They are so happy with your work on the Bird detector that they also hire you to build a Cat detector. (Wow Cat detectors are just incredibly useful aren’t they.) Because of years of working on Cat detectors, you have such a huge dataset of 100,000,000 cat images that training on this data takes about two weeks. Which of the statements do you agree with? (Check all that agree.)

- Buying faster computers could speed up your teams’ iteration speed and thus your team’s productivity.
- Needing two weeks to train will limit the speed at which you can iterate.
- If 100,000,000 examples is enough to build a good enough Cat detector, you might be better of training with just 10,000,000 examples to gain a \approx≈10x improvement in how quickly you can run experiments, even if each model performs a bit worse because it’s trained on less data.
- Having built a good Bird detector, you should be able to take the same model and hyperparameters and just apply it to the Cat dataset, so there is no need to iterate.

正确答案：第1、2、3项。不同问题的模型参数一般而言是不能直接照搬使用的。



	
