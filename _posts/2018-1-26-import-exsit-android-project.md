---
layout:     post
title:      Android studio导入项目注意事项
keywords:   博客
categories: [Android]
tags:	    [Android studio,导入项目]
---
Android studio导入项目时，由于插件版本不一致，需要进行一定的修改才能正常运行导入的项目。    
   
## Project的build.gradle   
我的android studio版本是3.0.1，它使用了google自家的maven仓库https://maven.google.com（如果因为局域网连接补上，可以在gradle.properties中添加代理），导入的项目如果使用的是studio3.0以前的版本，就要加上google()。另外，还需要修改classpath声明的gradle插件版本号。因为gradle不是专门为构建Android项目而开发的，想要用它来构建Android项目，则需要声明这个插件。

    buildscript {  
        repositories {
            google() //add
            jcenter()
        }
        dependencies {
            classpath 'com.android.tools.build:gradle:3.0.1' //modify

        }
    }

    allprojects {
        repositories {
            google() //add
            jcenter()
    }
   
   
     
## app的build.gradle
android闭包里的conmpileSdkVersion用于制定项目的编译脚本，这里26表示使用Android8.0.0系统的SDK编译。android闭包里又嵌套了一个defaultConfig闭包，其中可以对项目的更多细节进行配置，其中minSdkVersion用于指定项目最低兼容的Android系统版本，这里21表示最低兼容到Android5.0.1。targetSdkVersion指定的值表示在该目标版本上做过充分的测试。dependencies闭包可以指定当前项目所有的依赖关系。Android stuidio3.0以后，默认的依赖由之前的compile变为implementation了，compile指令被标注为过时方法。同时testCompile也变为testImplementation了。

    compileSdkVersion 26 //modify

    minSdkVersion 21    //modify
    targetSdkVersion 26 //modify

    dependencies {
        implementation fileTree(dir: 'libs', include: ['*.jar'])
        implementation 'com.android.support:appcompat-v7:26.1.0'  //modify
        implementation 'com.android.support.constraint:constraint-layout:1.0.2'  //modify
        testImplementation 'junit:junit:4.12'
        androidTestImplementation 'com.android.support.test:runner:1.0.1'
        androidTestImplementation 'com.android.support.test.espresso:espresso-core:3.0.1'
    }
         
  
    
## gradle的gradle-wrapper.properties
distributionUrl是要下载的gradle的地址，使用哪个版本的gradle，就在这里修改。
   
    distributionUrl=https\://services.gradle.org/distributions/gradle-4.1-all.zip //modify
  

