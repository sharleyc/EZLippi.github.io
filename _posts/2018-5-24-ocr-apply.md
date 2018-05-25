---
layout:     post
title:      AI在UI自动化测试中的应用尝试
keywords:   博客
categories: [Python]
tags:       [AI, UI自动化]
---

UI自动化测试的核心是控件的识别和操作，目前主流的识别是基于控件的遍历实现的，如果缺少目标控件的相关信息，就无法识别到控件了。基于图像识别的技术，由于其结果的准确性，影响了脚本的稳定性和可维护性，并没有得到广泛的应用。随着AI时代的来临，图像识别技术有了更快速的发展，新技术的发展和成熟，使得我们可以重新考虑图像识别技术在测试领域应用的可能。本文探讨了OCR识别在UI自动化测试中的应用实践。

## 接入OCR识别      

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

首先要拼接包头，里面包含前面提到的签名，然后获取图片(支持本地图片和url)进行base64编码，使用requests发送post请求，收到回包，回包里包含了识别到的文本内容及其位置信息。   

##  获得指定文字内容的位置信息     

遍历ocr返回的str中的itemstring，进行两个str的匹配，考虑ocr的识别准确性，允许设置最小匹配长度（即允许部分匹配），提高获得结果的成功率。在Python3里进行字符串匹配时需要注意str的编码，sdk返回的str的编码是iso-8859-1，需要转换成utf-8，才能正常匹配中文字符。

	def get_coord_by_ocr(text: str,ocr_result:str, min_hit:int = None)-> dict:

		if ocr_result is None:
			raise NameError('youtu.generalocr is None.')
		text_list = list(text)
		min_hit = min_hit if min_hit else len(text_list)
		text_coord = {}
		for item in ocr_result['items']:
			hint_count = 0
			item_string = item['itemstring'] 
            #sdk返回的str是iso-8859-1编码的，需要转化成utf-8编码
			item_string = item_string.encode('iso-8859-1').decode('utf-8')
			item_string_list = list(item_string)
			#计算匹配的字符数
			for char in text_list:
				if char in item_string_list:
					item_string_list.pop(item_string_list.index(char))
					hint_count +=1
			#匹配成功，返回字符串的位置信息
			if hint_count >= min_hit:
				text_coord = item['itemcoord']   
				return text_coord
		return text_coord   

##  操作控件     

前面两部分实际上完成了对控件的识别。接下来就是如何操作控件了。利用adb shell input tab可以模拟点击屏幕，生成down-up事件。通过获得的位置信息，可以计算出需要点击的x,y坐标的值，然后cmd执行adb shell input tab命令，就完成了对控件的点击操作。         
   
    x = x + width/2    
    y = y + height/2    
    
    adb shell input tab x y   


## 小结    

本文主要探讨了OCR识别技术在UI自动化测试中的实践应用。融合了ocr识别技术的UI自动化测试，提高了非Native场景下的测试能力，提升了脚本的稳定性。结合各种云测平台的真机资源，还可以将UI自动化测试应用于真机兼容性测试，进一步提升了UI自动化测试的ROI。







