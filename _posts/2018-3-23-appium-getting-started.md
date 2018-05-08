---
layout:     post
title:      Appium入门之安装与测试
keywords:   博客
categories: [Android,Appium]
tags:	    [自动化测试]
---

Appium自动化测试是值得学习和研究的技术。Appium的核心是一个暴露了REST API的网络服务器，这个服务器接收客户端过来的连接，监听客户端过来的命令，在移动设备上运行命令，然后把代表命令运行结果的HTTP响应包发送回客户端。其优点是能进行跨应用和跨平台测试，支持webview，支持设备无源码测试，支持多语言。缺点是不太稳定，比较难用。但是瑕不掩瑜，测试人员在选择移动端自动化测试框架时，Appium总是会在备选之列。学习和掌握Appium不仅能帮助我们根据项目需求更好的选择合适的自动化工具，也对开发自己的测试框架有借鉴意义。

## Appium原理

Appium是 C/S模式的。Appium是基于 webdriver 协议添加对移动设备自动化api扩展而成的。Webdriver 是基于 http协议的，第一连接会建立一个 session 会话，并通过 post发送一个 json 告知服务端相关测试信息。客户端只需要发送http请求实现通讯，意味着客户端就是多语言支持的。   

  ![](/images/images_2018/appium0.png)    

Appium的加载过程

1）调用Android adb完成基本的系统操作   
2）向Android上部署bootstrap.jar   
3）Bootstrap.jar Forward Android的端口到PC机器上   
4）PC上监听端口接收请求，使用webdriver协议    
5）分析命令并通过forward的端口转发给bootstrap.jar   
6）Bootstrap.jar接收请求并把命令发给uiautomator      
7）Uiautomator执行命令   

Bootstrap的作用   

Bootstrap是Uiautomator的测试脚本，它的入口类bootstrap继承于UiautomatorTestCase，所以Uiautomator可以正常运行它，它也可以正常使用uiautomator的方法，这个就是appium的命令可以转换成uiautomator命令的关键。   

Bootstrap还是一个socket服务器，专门监听4724端口过来的appium的连接和命令数据，并把appium的命令转换成uiautomator的命令来让uiautomator进行处理。  

因此Appium服务端既是服务端又是一个客户端。Appium启动时会创建一个http://127.0.0.1:4723/wd/hub服务端，脚本告诉服务端要做什么，服务端再去跟设备打交道，这里服务端相当于一个中转站。服务端和设备默认使用4724端口进行通讯，测试时服务端在设备上部署bootstrap.jar，启动这个包后会在手机上创建一个socket服务，对于socket服务来说，Appium服务端就是一个客户端。  


## Appium安装  

从第一部分Appium原理的介绍，我们能基本了解需要安装哪些：Android SDK、Appium服务端、Appium客户端。我的安装环境是win7(64位)系统。   

Step1 安装Android SDK以及设置环境变量   

网上资料一大堆，本文就不赘述了。已经有Android开发环境的可以直接跳到下一步。  

Step2 Appium-desktop下载和安装  

Appium Server and Inspector in Desktop GUIs for Mac, Windows, and Linux。根据官方描述，Appium-desktop是可以替代Appium Server的。下载地址：   
https://github.com/appium/appium-desktop/releases   
我下载的版本是appium-desktop-setup-1.4.0.exe，双击exe文件即可安装。安装完成后会在桌面生成一个紫色的Appium图标，双击打开：  

  ![](/images/images_2018/appium8.png)  
  
点击Start Server按钮启动服务：   

  ![](/images/images_2018/appium9.png)    

Step3 Appium-server安装   

安装了Appium-desktop，就不用安装了appium-server了。我两个都装了，所以这里都介绍一下安装过程。注意选择安装appium-server，而不是appium-desktop的话，还需要提前安装nodejs（官网https://nodejs.org/en/），我开始安装了一个最新的稳定版本，后面发现命令行方式无法启动appium，不得不把nodejs降级到一个低版本才解决问题。


我们可以在Appium官方网站上下载操作系统相应的Appium版本：   
https://bitbucket.org/appium/appium.app/downloads/   

我选择了AppiumForWindows_1_4_16_1.zip（相对比较稳定的一个版本）。解压zip文件后点击appium-installer.exe进行安装，安装路径：D:\Program Files (x86)\Appium，安装完成，需要将以下目录添加环境变量path下面：   
D:\Program Files (x86)\Appium\node_modules\.bin   
最后cmd窗口运行命令：appium-doctor，如果结果显示"All Checks were successful"，则说明Appium所需要的各项环境都已准备完成。  

  ![](/images/images_2018/appium1.png)     

Step4 Appium Java-client   

这里根据你选择的语言安装对应的appium-client，本文以Java为例。下载appium的jar客户端和selenium类库（appium继承了webdriver），在Android Studio导入使用。下载地址是： 
https://github.com/appium/java-client   
http://docs.seleniumhq.org/download/   
（这里省略安装Android studio的教程）  
  

  
## 使用Android Studio创建Appium实例    

Step1 打开AS（我的AS版本是3.0.1），因为AS不能创建Java项目，所以先创建一个android项目，再New一个Java library的module，我这里命名为demotest1。  

Step2 项目切换到project视图下，在demotest1-java文件夹上右键选择New package，命名为libs，将之前打包下载的两个jar包拷贝到libs文件夹下，然后分别选中这两个jar，右键选择Add as library，选择demotest1这个module，确定，等待编译完毕。 

  ![](/images/images_2018/appium5.png)    

Step3 编写测试脚本 
   
    package com.example;

    import org.openqa.selenium.remote.DesiredCapabilities;
    import org.testng.annotations.AfterMethod;
    import org.testng.annotations.AfterSuite;
    import org.testng.annotations.BeforeMethod;
    import org.testng.annotations.BeforeSuite;
    import org.testng.annotations.Test;

    import java.net.MalformedURLException;
    import java.net.URL;
    import java.util.concurrent.TimeUnit;
    import io.appium.java_client.android.AndroidDriver;

    public class SimpleTest {
        private AndroidDriver driver;

        @BeforeSuite
        public  void beforeSuite() throws MalformedURLException{

            //set up appium
            DesiredCapabilities capabilities = new DesiredCapabilities();
            capabilities.setCapability("automationName","Appium");
            capabilities.setCapability("deviceName","35eb3480");
            capabilities.setCapability("platformName","Android");
            capabilities.setCapability("platformVersion","5.1.1");
            capabilities.setCapability("appPackage","com.miui.calculator");
            capabilities.setCapability("appActivity","com.miui.calculator.cal.NormalCalculatorActivity");

            driver = new AndroidDriver(new URL("http://127.0.0.1:4723/wd/hub"),capabilities);
        }

        @AfterSuite
        public void afterSuite(){
            driver.quit();
        }

        @BeforeMethod
        public void beforeMethod() throws Exception{
            System.out.println("beforeMethod");
        }

        @AfterMethod
        public  void afterMethod() throws Exception{
            System.out.println("afterMethod");
        }

        @Test
        public void test1() throws Exception{
            driver.findElementById("btn_1").click();
            driver.findElementById("btn_plus").click();
            driver.findElementById("btn_2").click();
            driver.manage().timeouts().implicitlyWait(5, TimeUnit.SECONDS);
        }
    }

这里解释一下DesiredCapabilities的作用：负责启动服务端时的参数设置，启动session的时候是必须提供的。我的这个实例用的是真机，运行程序之前要把设备用USB数据线连到电脑上，cmd下运行adb devices查看设备的唯一标识符，对应的是deviceName的值，platformVersion的值则是设备真实的系统版本号。appPackage和appActivity的获取方式可以利用adb shell dumpsys命令，具体命令如下：  
adb shell dumpsys window windows | grep mCurrent   


  

## 参考资料 
https://blog.csdn.net/jffhy2017/article/details/69220719    
http://blog.csdn.net/niubitianping/article/details/52523239?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io    
https://www.cnblogs.com/wysk/p/7346659.html   
https://www.cnblogs.com/fnng/p/4540731.html    
  





