---
layout:     post
title:      收藏的测试文章列表（11/19/17 - 11/25/17）
keywords:   博客
categories: [收藏]
tags:	    [测试网站]
---

以下是本周（11/19/17 - 11/25/17）收藏的测试文章列表。本周的主题是：Testing shifts to production。   

* Testing the Unexpected: A Shift Right in DevOps Testing   

https://www.stickyminds.com/article/testing-unexpected-shift-right-devops-testing    

关注词：Diff，canary release，Chaos Monkey   


* Testing and monitoring in production - your QA is incomplete without it     

https://assertible.com/blog/testing-and-monitoring-in-production-your-qa-is-incomplete-without-it   

一个彻底的测试计划将优先考虑生产监控作为对生产前测试和传统自动化测试的补充。  

测试人员可能会关注如何快速地发布未经完全测试的功能，但可以使用像金丝雀释放测试（Canary Release Testing）和曝光控制（Exposure Control）这样的发布方法来缓慢推出更改。   

记住：测试不是关于错误报告，而是关于质量。  

通过更快的发布周期，测试目标不再试图完全根除故障，而是建立可以快速识别和提醒问题的可恢复系统。 这意味着生产API的监测和自动化测试。   

了解在测试计划中可以集成TiP和MiP（生产中的测试和监控）的不同方式   

* 7 WAYS TO BUILD A ROBUST TESTING IN PRODUCTION (TIP) PRACTICE   

https://www.neotys.com/blog/7-ways-to-build-a-robust-testing-in-production-practice/   

任何软件测试人员都应该了解许多不同类型的TiP：    

Canary Testing   
Controlled Test Flight   
A/B Split Testing   
Synthetic User Testing   
Fault Injection（Chaos Monkey ）   
Recovery Testing   
Data Driven Quality   

确保测试人员能够访问生产环境中的日志，性能指标，警报和其他信息，以便他们能够主动识别和解决问题。   

* DON’T DO IT THE WRONG WAY: TIPS FOR TESTING IN PRODUCTION   

https://www.neotys.com/blog/tips-for-testing-in-production/   

对生产测试进行分层，以尽量减少测试和维护测试环境对生产用户的影响。   

当运行生产测试时，请注意关键的用户性能指标，以便知道测试是否对用户体验产生任何不可接受的影响。如果是这样，准备关闭测试。   

* 5 WAYS SYNTHETIC USERS MAKE YOUR WEBSITE MONITORING SMARTER   

https://www.neotys.com/blog/5-ways-synthetic-users-make-website-monitoring-smarter/   

Scala, Scalatest, 95%--99.5% reliability    

* Diffy: Testing services without writing tests   

https://blog.twitter.com/engineering/en_us/a/2015/diffy-testing-services-without-writing-tests.html    

Diffy的前提是，如果服务的两个实现返回一个足够大和不同的请求“相似”的响应，那么这两个实现可以被视为等价，新的实现是免回归的。   
我们使用“相似”而不是“相同”的，因为响应可能会产生大量的噪声，使响应数据结构的某些部分不确定。 例如：   
嵌入在响应中的服务器生成的时间戳   
在代码中使用随机生成器   
下游服务提供的实时数据中的竞争条件   

Diffy的新型噪声消除技术与其他基于比较的回归分析工具不同。   
Diffy的工作原理:   
Diffy作为一个代理接受来自您提供的任何源的请求，并将这些请求中的每个请求发送到三个不同的服务实例：
运行新代码的候选实例   
运行最后一个已知正确的代码的主要实例   
与主实例运行相同的已知良好的代码的辅助实例   

当这些服务发回响应时，Diffy比较这些响应并寻找两件事情：   
候选实例和主要实例之间观察到的原始差异。   
在主要和次要实例之间观察到的非确定性噪声。    
由于这两个实例都运行在已知良好的代码上，所以我们理想地期望响应是相同的。    然而，对于大多数真实的服务，我们观察到一些部分的反应最终是不同的，并表现出不确定的行为。    

Diffy衡量主要和次要实例彼此不一致的频率，与主要和候选实例彼此不一致的频率。如果两个频率大致相同，则确定没有任何错误，并且可以忽略该错误。




  
