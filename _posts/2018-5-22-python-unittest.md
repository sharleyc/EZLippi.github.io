---
layout:     post
title:      Python单元测试框架-unittest
keywords:   博客
categories: [Python]
tags:	    [unittest]
---

unittest是python的单元测试框架，是由java的JUnit衍生而来。unittest最核心的四个概念是test case, test suite, test runner, test fixture。这里借用网络上的一张图：

  ![](/images/images_2018/unittest.png)

## test case  

A test case is the smallest unit of testing. It checks for a specific response to a particular set of inputs. unittest provides a base class, TestCase, which may be used to create new test cases.

一个test case是测试的最小单元。unittest提供了一个基类TestCase，通过实例化TestCase类，就能得到新的test cases。一个test case对应一个完整的测试流程，通常以test开头，因为如果使用defaultTestLoader，会默认把test开头的方法识别为一个用例（见loader.py文件）。  
    defaultTestLoader = TestLoader()   
    ......   
    testMethodPrefix = 'test'   
    return attrname.startswith(prefix) and hasattr(getattr(testCaseClass, attrname), '\_\_call\_\_')
  
以下列出来一些TestCase类里常用的方法(见case.py文件)：  
1、setUp(): 在每个test case执行前都要执行的方法。  
2、tearDown(): 在每个test case执行后都要执行的方法。（不管成功与否）  
3、setUpClass(): 类方法，所有test cases开始之前，执行一次。   
4、tearDownClass(): 类方法，所有test cases结束之后，执行一次。    
5、run(result): 测试执行完毕后，测试结果存储在result中。
示例代码 class TestMathFunc(unittest.TestCase): 
    
    def setUp(self):
        print "do something before test. Prepare environment."
    
    def tearDown(self):
        print "do something after test. Clean up."
    
    def test_add(self):
        '''Test method add(a,b)'''
        self.assertEqual(3,add(1,2))
        self.assertNotEqual(3,add(2,2))
        
    def test_minus(self):
        '''Test method minus(a,b)'''
        self.assertEqual(1,minus(3,2))
        
    def test_multi(self):
        '''Test method multi(a,b)'''
        self.assertEqual(6,multi(2,3))
    
    def test_divide(self):
        '''Test method divide(a,b)'''
        self.assertEqual(2,divide(6,3))

TestCase子类中的所有测试用例都是独立的，通过使用类属性，还是能共享数据的。

## test suite   

A test suite is a collection of test cases, test suites, or both. It is used to aggregate tests that should be executed together. 
 
多个test cases组合在一起就是test suite，test suite也可以再嵌套test suite。添加到test suite的test case是会按照添加的顺序执行的。部分源码如下： 

    suite = unittest.TestSuite()
    
    tests = [TestMathFunc("test_add"),TestMathFunc("test_minus"),TestMathFunc("test_divide")]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
 

TestSuite继承于BaseTestSuite（见suite.py文件），其主要属性和方法如下：   
1、self._tests: 这个私有变量里包含所有的test cases或test suite。  
2、run(result): 该方法遍历self._tests变量，最终所有的执行都是执行TestCase中的run()，测试结果存储在result中。

## test runner  

A test runner is a component which orchestrates the execution of tests and provides the outcome to the user. The runner may use a graphical interface, a textual interface, or return a special value to indicate the results of executing the tests.  

test runner协调测试的执行并向用户提供结果（见runner.py文件）：  
1、TextTestResult类: 储存测试结果的类，继承于TestRult类。调用run方法时，必须传一个result，这个result就是TestResult类或其子类的对象，通过调用addSuccess，addError，addFailure，addSkip等方法将执行结果保存在实例化的对象属性中。  
2、TextTestRunner类: 以文本形式显示测试结果的类。一开始就使用了TextTestResult类，最终把TextTestResult实例对象传递给TestCase的run方法。  

其中verbosity参数可以控制执行结果的输出：0是静默模式，只能获得简单报告；1是默认模式，类似静默模式，成功的用例前面有个“.”，失败的用例前面有个“F”，比如依次执行三个用例，如果返回..F，则表示第1，2个用例是成功的，第3个用例是失败的；2是详细模式，可以获得详细报告，测试结果会显示用例名称等详细信息。



##  test fixture  

A test fixture represents the preparation needed to perform one or more tests, and any associate cleanup actions. This may involve, for example, creating temporary or proxy databases, directories, or starting a server process.  

test fixture，简单的说，是测试运行之前所需的稳定的、公共的可重复的运行环境。这个环境不仅可以是数据，也可以指对被测软件的准备，例如创建临时的数据库、启动服务进程等。text fixture的主要目的是建立一个固定或已知的环境状态来确保测试可重复并且按照预期方式运行。其中setUp()和tearDown()是最常用的方法。   

参考资料：   
https://blog.csdn.net/huilan_same/article/details/52944782    






