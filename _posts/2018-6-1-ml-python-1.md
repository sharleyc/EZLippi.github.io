---
layout:     post
title:      Machine Learning in Python：使用scikit-learn预测Google股票
keywords:   博客
categories: [Python]
tags:	    [scikit-learn，线性回归]
---

前面我们已经学习了机器学习的线性回归模型，但是纸上得来终觉浅，我们需要亲身实践一下才能获得更深刻的认识。

我们先简单回顾一下线性回归：简单的说，就是要寻找最适合数据的线性方程，并能对特定值进行预测。由于回归广泛用于预测股票价格，我们使用线性回归来预测Google的股票价格（请注意这只是一个示例，不适用于炒股）。总体来说，我们需要完成以下四个步骤：1、获取数据；2、确定特征和标签；3、训练和测试；4、预测。    


## 准备工作  

准备工作主要是安装Python开发环境和相关库。笔者使用的Python3.6版本，使用的数据来自www.quandl.com网站，所以需要安装quandl库，此外还需要安装pandas,numpy以及sklearn库。为了不受限制的使用quandl的免费资源，建议注册一个帐号，获取专用的API Key。 

    import pandas as pd
    import quandl,math
    import numpy as np
    from sklearn import preprocessing, svm
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    from matplotlib import style  
    from datetime import datetime

## 获取数据           

获取数据的方法很简单，以下两行代码就能获得GOOGL股票的价格数据。   

    quandl.ApiConfig.api_key = "E5Xw3oe******HwY86hd"
    df = quandl.get('WIKI/GOOGL')     


## 确定特征和标签      

我们需要预测的是股票的价格，对应的是表格中的Adj. Close字段，我们先查看一下Adj. Close的变化趋势，从2016年到2018年基本为线性增长：      

    df['Adj. Close'].plot()
    plt.show()  

  ![](/images/images_2018/6-1_0.png)    

原始的DataFrame有很多列，但有些是多余的。我们删减原始的DataFrame，保留需要的数据，在这个示例中，只保留了开盘价、最高价、最低价、收盘价和成交量。同时，原始数据需要进行一些转换，使其更有价值，才能提供给机器学习算法。因此，'Adj. Close','HL_PCT','PCT_change','Adj. Volume'才是真正用到的特征字段。       

    df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
    df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
    print(df.head())       

  ![](/images/images_2018/6-1_1.png)   

我们要预测的是未来某个点的价格，因此我们添加了预测列，并将所有的NaN数据填充为-99999。在机器学习中，对于如何处理缺失数据，一个主流选项就是将缺失值填充为-99999（因为sklearn不会处理空数据，需要把空数据设置为一个比较难出现的值）。我们还需要指定预测的数量，如果我们的数据是100天的股票价格，那么我们需要预测未来一天的价格，即预测整个数据长度的1%。我们用label代表预测结果字段，通过让Adj. Close列的数据往前移动1%行来表示。最后生成真正在模型中使用的数据X和y。至此，我们有了数据，包含特征和标签。
  
       
    forecast_col = 'Adj. Close' 
    df.fillna(-99999,inplace=True)
    forecast_out = int(math.ceil(0.01*len(df)))  
    df['label'] = df[forecast_col].shift(-forecast_out)  
    print(df.tail())

  ![](/images/images_2018/6-1_2.png)   

在机器学习里通常把X作为特征，y作为特征的标签，我们可以这样定义我们的特征和标签。使用drop方法删除label列，返回一个新的DataFrame，并转换成numpy数组。在训练和测试之前，需要对数据做一些预处理。preprocessing.scale(X)的计算方式是将特征值减去均值，除以标准差（所有数减去平均值的平方和，除以该组数之个数，再开根号），变换后各维特征有0均值，单位方差，也叫零均值标准化。标准化通常会加速处理过程，有助于提高准确性。   


    X = np.array(df.drop(['label'],1))
    X = preprocessing.scale(X) 

提取历史数据的前99%作为训练集和测试集，后1%用来进行预测。因此需要去掉label列中为空的那些行。设置label字段为目标值y。至此，我们得到了特征X和标签y。     

    X = X[:-forecast_out]
    X_lately = X[-forecast_out:]

    df.dropna(inplace=True)
    y = np.array(df['label'])   

## 训练和测试 
 
上面我们已经准备好了数据，可以开始构建模型并训练它。使用train_test_split可以打乱数据，选取20%的数据作为测试集，80%的数据作为训练集。Sklearn提供了许多通用的分类器，我们选择了linear_model中的LinearRegression作为分类器后，就可以使用fit来训练了。通过fit拟合我们的训练特征和标签，然后使用score加载测试，最后打印出准确率。也许我们会尝试大量的算法并且仅仅选取最好的那个。有些算法必须线性运行，有些则不是。对于后者，我们可以使用多线程来获取更高的性能。  

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    clf = LinearRegression() 
    #clf = LinearRegression(n_jobs=-1) 
    #clf = svm.SVR() # 0.80
    #clf = svm.SVR(kernel='poly')#0.66
    clf.fit(X_train,y_train)
    accuracy = clf.score(X_test,y_test)  
    print(accuracy) #0.97

    
## 预测结果并可视化       

我们将用上一步中获得的模型来进行预测，预测结果放在forecast_set中。         

    forecast_set = clf.predict(X_lately)
    print(forecast_set,accuracy,forecast_out)

   ![](/images/images_2018/6-1_3.png)    
 
最后我们需要将预测值和历史数据拼接起来，并展示可视化结果。首先在历史数据表格中单独创建一个名为Forecast的字段，用来存放预测结果：  

    df['Forecast'] = np.nan   

获取历史数据表中最后一天的日期信息，并转换为秒的格式。设置一天为86400秒，下一天的计算方法则为当前日期数值加上86400秒：     

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day   

循环将预测结果写入历史数据表中：     

    for i in forecast_set:
        next_date = datetime.fromtimestamp(next_unix)
        next_unix += one_day
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]  

分别查看预测前和预测后的数据表：     
  

   ![](/images/images_2018/6-1_4.png)    

使用图表对历史数据和预测数据进行拼接和可视化：   

    df['Adj. Close'].plot()
    df['Forecast'].plot()
    plt.legend(loc=4)  
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

   ![](/images/images_2018/6-1_5.png)   
