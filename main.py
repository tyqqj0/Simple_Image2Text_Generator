# -*- CODING: UTF-8 -*-
# @time 2024/6/14 下午3:30
# @Author tyqqj
# @File main.py
# @
# @Aim
import time

import numpy as np
import torch
from torch.utils import data
from torch import nn

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

from zhipuai import ZhipuAI

import numpy as np
from PIL import Image
import io
import oss2

taccess_key_id = ''
taccess_key_secret = ''
tbucket_name = ''
tendpoint = 'oss-cn-beijing.aliyuncs.com'


class AliyunOSSImageHost:
    def __init__(self, access_key_id=None, access_key_secret=None, bucket_name=None, endpoint=None):
        if access_key_id is None:
            access_key_id = taccess_key_id
        if access_key_secret is None:
            access_key_secret = taccess_key_secret
        if bucket_name is None:
            bucket_name = tbucket_name
        if endpoint is None:
            endpoint = tendpoint

        self.auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(self.auth, endpoint, bucket_name)

    def upload_image(self, image_path):
        print("Uploading image to Aliyun OSS...")
        file_key = image_path.split('/')[-1]
        result = self.bucket.put_object_from_file(file_key, image_path)
        # 假设原始的endpoint可能包含'http://'
        if 'http://' in self.bucket.endpoint:
            bucket_endpoint = self.bucket.endpoint.replace('http://', '')

        # 或者如果包含'https://'
        elif 'https://' in self.bucket.endpoint:
            bucket_endpoint = self.bucket.endpoint.replace('https://', '')
        if result.status == 200:
            return f"http://{self.bucket.bucket_name}.{bucket_endpoint}/{file_key}"
        else:
            return None

    def upload_numpy_array(self, array: np.array, file_name=None):
        """
        将NumPy数组转换为图像并上传到OSS。
        :param array: NumPy二维数组
        :param file_name: 保存在OSS上的文件名
        :return: 图片在OSS上的URL或上传失败时返回None
        """
        print("Uploading image to Aliyun OSS...")
        if file_name is None:
            timett = str(int(time.time()))
            file_name = f"numpy_array_{timett}.png"
        # 确保数组是二维的
        if array.ndim != 2:
            raise ValueError("Only 2D arrays are supported.")

        # 将NumPy数组转换为Pillow图像
        image = Image.fromarray(np.uint8(array))

        # 将图像保存到内存中的文件对象
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # 上传到OSS
        result = self.bucket.put_object(file_name, img_byte_arr)
        # 假设原始的endpoint可能包含'http://'
        if 'http://' in self.bucket.endpoint:
            bucket_endpoint = self.bucket.endpoint.replace('http://', '')

        # 或者如果包含'https://'
        elif 'https://' in self.bucket.endpoint:
            bucket_endpoint = self.bucket.endpoint.replace('https://', '')
        if result.status == 200:
            return f"http://{self.bucket.bucket_name}.{bucket_endpoint}/{file_name}"
        else:
            return None







class ImageTextGenerator:
    def __init__(self, api_key=None, prompt=None):
        if prompt is None:
            self.prompt = ("请分析如下脑部血管的医学图像,并从以下三个角度撰写一份图像描述报告:"
                           "\n图像基本特征:描述该图像所展示的器官或组织结构,以及图像中出现的明显视觉特征,如亮度、形状、纹理等。请根据医学先验知识判断这些特征是否符合解剖学常识。"
                           "\n标注质量:仔细观察图像中结构的勾画和标注情况,评估标注的准确性、连续性和完整性。如果发现标注存在问题,请具体指出。"
                           "\n需要关注的部分:基于解剖学知识,判断图像中是否存在应该出现但未被明确表示的重要结构。如果发现任何异常或可疑区域,请重点描述其位置和表现。"
                           "\n请注意:你的描述必须紧扣所给图像的实际内容, 不得臆测或虚构任何信息。每个角度的分析都要有理有据, 尽可能地利用医学专业知识来支撑你的观点。你的报告应该切中肯綮、简明扼要, 突出需要关注的重点问题, 避免冗长的描述。"
                           "明白了吗?明白的话, 请分析以下图像:")
        if api_key is None:
            api_key = "11674de81b52a244985b70ad1dc9873f.C6wucI9fuZzgbJuZ"  # "11674de81
        self.client = ZhipuAI(api_key=api_key)
        self.image_host = AliyunOSSImageHost()

    def generate_text(self, image_url):
        # while True:
        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="glm-4v",  # 填写需要调用的模型名称,
                    messages=[
                        {"role": "user",
                         "content": [
                             {"type": "text",
                              "text": self.prompt
                              },
                             {"type": "image_url",
                              "image_url": {"url": image_url}
                              }
                         ]
                         }
                    ]
                )
                if response.choices[0].message.content == "抱歉，我目前还没有修改图片的能力。如果您有其他请求，欢迎随时向我提问。谢谢！":
                    print("Retrying...")
                    pass
                else:
                    # break
                    return response.choices[0].message.content
            except Exception as e:
                print(image_url)
                print(e)
                time.sleep(0.1) # 等待1秒后重试
        print("Failed to generate text, please try a different image or prompt.")
        return None


    def generate_text_from_image(self, image_path):
        image_url = self.image_host.upload_image(image_path)
        return self.generate_text(image_url)

    def generate_text_from_numpy_array(self, array: np.array):
        image_url = self.image_host.upload_numpy_array(array)
        return self.generate_text(image_url)







if __name__ == '__main__':

    image_text_generator = ImageTextGenerator()
    image_path = "Snipaste_2024-06-09_13-27-16.png"
    image_path2 = "Snipaste_2024-06-14_20-43-51.png"
    text = image_text_generator.generate_text_from_image(image_path2)
    print(text)
    print("Done!")
    if False:
        client = ZhipuAI(api_key="11674de81b52a244985b70ad1dc9873f.C6wucI9fuZzgbJuZ")  # 填写您自己的APIKey
        # img = "https://pic.imgdb.cn/item/666bf289d9c307b7e99131a7.png"
        # img1 = "https://pic.imgdb.cn/item/666c0067d9c307b7e9aaef09.png"
        img_file = "./Snipaste_2024-06-09_13-27-16.png"
        # 读取
        # img_file = Image.open(img_file)
        img = AliyunOSSImageHost().upload_image(img_file)
        print(img)
        # img = "https://image-tyqqj.oss-cn-beijing.aliyuncs.com/Snipaste_2024-06-09_13-27-16.png?Expires=1718367654&OSSAccessKeyId=TMP.3KgG7Vp9exHntsexvhEDtpxRcQVfMEyhnpsyMZoigRnRAP4UJftUF81mHnWxgpC8r4yNNjeKG94Y2Whs6zsKeyVKjaZBVc&Signature=dBuMs5ZPrrXfv0MbB246G65Hwss%3D"
        text = text1 = "请评价一下这张脑部血管成像的质量, 你认为这张成像的质量如何?"
        response = client.chat.completions.create(
            model="glm-4v",  # 填写需要调用的模型名称,
            # temperature=0.1,
            messages=[
                {"role": "user",
                 "content": [
                     {"type": "text",
                      "text": text
                      },
                     {"type": "image_url",
                      "image_url": {"url": img}
                      }
                 ]
                 }
                # },
            ]
        )
        print(response.choices[0].message.content)