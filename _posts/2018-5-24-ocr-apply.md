---
layout:     post
title:      AI在UI自动化测试中的应用尝试
keywords:   博客
categories: [Python]
tags:       [AI, UI自动化]
---

UI自动化测试中最核心的技术之一是控件识别，目前主流的控件识别是基于控件遍历实现的，如果缺少目标控件的相关信息，就无法识别到控件了。UI自动化测试的构建和维护成本都限制了其应用范围和效果。现在全民都在迎接AI时代，AI在测试领域也开始得到应用。本文探讨了OCR识别在UI自动化测试的控件识别中的应用。

## OCR识别      

腾讯优图-AI开放平台免费对外开放了众多AI功能的API，其中就包括OCR识别。利用其中的通用印刷体文字识别功能，可以获取到图片中的指定文字内容的位置，轻松完成控件识别任务。具体实现流程如下：      
    

1、 申请AppID,SecretID,SecretKey   

AI开发平台的功能虽然是免费的，但必须使用开发密钥初始化。参考新手指引-免费接入优图开放平台，即可获得开发密钥。（申请提交后是自动审批的）
   
2、下载SDK，调试sample  

笔者使用的脚本是Python3，所以下载的是Python版本的sdk（Python_sdk_master.zip），解压后进入目录，执行python setup.py install完成安装。打开sample.py，找到以下代码：       
    #general ocr: use local image   
    #retgeneralocr = youtu.generalocr('icon_ocr_common01.png', data_type = 0)    
    #print retgeneralocr   
修改后的代码如下：    
    #general ocr: use local image    
    retgeneralocr = youtu.generalocr('icon_ocr_common01.png', data_type = 0)    
    print(json.dumps(retgeneralocr,indent=4))     
并在首部引入json模块：   
    import json    
注意：python3中print ***是不允许的，改成了print()。    

3、源码解析    

我们只用看auth.py和youtu.py这两个文件。auth.py里只定义了Auth类，它的app_sign()方法主要是用于生成签名，我们可以不用太关注签名算法的细节，除非想自己实现签名算法。重点是youtu.py文件，直接看926行，generalocr()方法。   

    def generalocr(self, image_path, data_type = 0, seq = ''):

        req_type='generalocr'
        headers = self.get_headers(req_type)
        url = self.generate_res_url(req_type, 2)
        data = {
            "app_id": self._appid,
            "session_id": seq,
        }

        if len(image_path) == 0:
            return {'httpcode':0, 'errorcode':self.IMAGE_PATH_EMPTY, 'errormsg':'IMAGE_PATH_EMPTY'}

        if data_type == 0:
            filepath = os.path.abspath(image_path)
            if not os.path.exists(filepath):
                return {'httpcode':0, 'errorcode':self.IMAGE_FILE_NOT_EXISTS, 'errormsg':'IMAGE_FILE_NOT_EXISTS'}

            data["image"] = base64.b64encode(open(filepath, 'rb').read()).rstrip().decode('utf-8')
        else:
            data["url"] = image_path

        r = {}
        try:
            r = requests.post(url, headers=headers, data = json.dumps(data))
            if r.status_code != 200:
                return {'httpcode':r.status_code, 'errorcode':'', 'errormsg':''}

            ret = r.json()
        except Exception as e:
            return {'httpcode':0, 'errorcode':self.IMAGE_NETWORK_ERROR, 'errormsg':str(e)}

        return ret     

首先要拼接包头，里面包含前面提到的签名，然后获取图片(支持本地图片和url)进行base64编码，使用requests发送post请求，收到回包。   

##  OCR识别给UI自动化测试赋能   








