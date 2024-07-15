# -*- CODING: UTF-8 -*-
# @time 2024/6/14 下午3:30
# @Author tyqqj
# @File Agent.py
# @
# @Aim
import json
import os
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

# taccess_key_id = ''
# taccess_key_secret = ''
# tbucket_name = ''
# tendpoint = 'oss-cn-beijing.aliyuncs.com'

# 获取当前的文件位置
# print(os.path.abspath(__file__))
file_path = os.path.abspath(__file__)
oss_dict = json.load(open(os.path.join(os.path.dirname(file_path), "oss_key.json"), "r"))
taccess_key_id = oss_dict["taccess_key_id"]
taccess_key_secret = oss_dict["taccess_key_secret"]
tbucket_name = oss_dict["tbucket_name"]
tendpoint = oss_dict["tendpoint"]

message = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": ("请分析如下医学图像,并按照以下三个角度撰写一份图像描述报告:\n"
                         "图像基本特征:首先说明该图像的有效区域主要位置(右侧，左中，左上，全部，中心区域)，然后描述该图像所展示的脑部器官或组织结构,以及图像中出现的明显视觉特征,如亮度、形状、纹理等。请根据医学先验知识判断这些特征是否符合解剖学常识"
                         "成像质量:仔细观察图像中结构的成像情况,评估成像的准确性、连续性和完整性，尤其时成像的关键部位边界是否模糊。如果发现标注存在问题,请具体指出。\n"
                         "需要关注的部分:基于解剖学知识以及成像质量,判断图像中是否存在应该出现但未被明确表示的重要结构，或成像质量不佳需要特殊注意的部分，如(边缘血管成像较为模糊，标注时注意血管连接性)。如果发现任何异常或可疑区域,请重点描述其位置和表现。\n"
                         "请注意:"
                         "你的描述必须紧扣所给图像的实际内容,不得臆测或虚构任何信息。\n"
                         "每个角度的分析都要有理有据,尽可能地利用医学专业知识来支撑你的观点。\n"
                         "你的报告应该切中肯綮、简明扼要,突出需要关注的重点问题,避免冗长的描述。\n"
                         "明白了吗?明白的话,请分析以下图像:\n")
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://image-tyqqj.oss-cn-beijing.aliyuncs.com/Normal002_image_2%281%29.png"
                }
            }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "图像基本特征:\n"
                        "该图像中心，右侧，下侧可以看见有效的图像。\n该图像展示了一张荧光显微镜下的细胞图像。视野中可见多个高亮的斑块状结构,推测为细胞核的荧光标记。细胞核形态大小不一,多呈圆形或椭圆形,分布较为零散。除细胞核外,背景区域荧光信号较弱,细胞质和其他细胞器结构不明显。整体而言,荧光强度分布不均匀,提示标记效率可能有待优化。\n"
                        "标注质量:\n"
                        "通过仔细观察发现,该图像中并未对任何细胞结构进行标注勾画。所有的高亮斑块均为荧光标记本身,并非人为标注结果。因此无法评估标注的准确性和完整度。如果该图像旨在分析特定细胞结构,建议后期进行必要的标注和注释,以便读者理解。\n"
                        "需要关注的部分:\n"
                        "虽然细胞核标记较为明显,但细胞核形态大小差异较大,提示细胞处于不同的生长状态,可能出现了异质性。同时,部分区域细胞核分布密度较低(如左下角区域),原因可能是标本制备过程中细胞脱落,也可能提示这些区域细胞生长状态异常。建议制备多个重复样本,评估细胞生长的整体情况。\n"
                        "此外,由于缺乏其他细胞结构的染色,很难判断细胞的完整性和形态特征。如有必要,可以考虑进行多重荧光标记,以同时观察细胞膜、细胞质等结构,全面评估细胞状态。"
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "这个分析存在很大的问题，这不一张细胞图像而是脑血管MRA医学图像的成像图，你在按照上述的报告结构分析的同时最好有体现到这些与MRA脑血管成像相关的关键词对提供的医学图像进行分析 关键词:\n"
                        "亮度 脑血管中心区域呈现高亮度\n"
                        "形状 脑血管呈现管状结构\n"
                        "整体走向 脑血管呈现曲折的走向\n"
                        "血管连续性 脑血管分支之间保持良好的连续性\n"
                        "切片 在不同切片上观察到的脑血管结构保持一致，显示出切片间的显著关联性\n"
                        "密度 脑血管的密度分布呈现不均匀性\n 下面请分析这张图"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://image-tyqqj.oss-cn-beijing.aliyuncs.com/Normal002_image_4%282%29.png"
                }
            }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "图像基本特征:\n"
                        "该图像在全部位置成像效果较好。\n该图像展示了脑血管的MRA渲染结果。整体上,血管呈现出清晰的管状结构,管腔内部亮度较高。血管主干粗大,分支逐级变细,整体呈现出典型的树状分支形态。血管在空间上走行迂曲,但总体连续性良好,未见明显中断。不同切片层面上观察到的血管形态保持一致,反映了血管网络的三维延续性。\n"
                        "标注质量:\n"
                        "图像中血管结构勾画清晰完整,主要血管分支均得以展现,未见明显遗漏或错误标注。三维渲染技术很好地再现了血管网络的立体形态,使得不同切面血管之间的关联性得以直观呈现。然而,由于图像整体亮度较高,部分细微血管分支的识别度稍显不足,可能会影响对血管末梢形态的精细刻画。\n"
                        "需要关注的部分:\n"
                        "通过观察发现,该脑血管图像在不同区域呈现出密度分布不均的特点。具体表现为:左侧部分区域血管密度较高,而右侧后部血管密度相对较低。这种区域性差异在一定程度上反映了脑血管网络的自然特点,但如果差异过于显著,则需要警惕是否存在局部血管发育异常或病变的可能。此外,中心区域散见管径不均的血管,其生理意义有待进一步评估,需要结合临床和血流动力学信息加以综合分析。\n"
                        "总之,该图像通过三维渲染技术较好地展示了脑血管的整体形态特征,血管标注质量总体满意。但局部血管密度分布的差异性以及管径变化的病理意义仍需深入探讨,以期为临床诊疗提供更有价值的参考信息。"

            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "你的分析存在问题，这张图片的成像效果实际并不是那么的好，首先这张图像仅在中偏右下有有效图像，中间有一条血管有明显的成像不均匀的断裂，注意中间的血管往右侧延伸时并没有连接到右下角主要的动脉血管中。如果出现这种情况，你需要在需要注意的地方着重强调。\n"
                        "注意需要你分析的图片的质量不一定是很好的图片，你的关键任务是从以下这些关键词指示的方向中判断图像质量，并严格按照要求进行合理地分析\n"
                        "请分析下面这个图片"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://image-tyqqj.oss-cn-beijing.aliyuncs.com/Normal002_image_5%283%29.png"
                }
            }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "图像基本特征:\n"
                        "该图像在中偏左下部分有有效图像。\n该图像展示了血管造影成像的结果。可见一条主要血管干及其分支血管的形态,血管呈现高亮度的树枝状分布。主干血管管径较粗,分支血管逐渐变细。血管走行比较平直,未见明显扭曲。总体上血管形态、分支模式与正常解剖结构相符。\n"
                        "标注质量:\n"
                        "图中主干血管及一级分支的勾画比较完整和准确,但二级以下的细小分支血管标注略有缺失,局部血管末梢连续性稍差。血管边缘的标注也不甚平滑。总的来说,关键结构标注到位,但在细节刻画上有进一步优化的空间。\n"
                        "需要关注的部分:\n"
                        "虽然血管主干和分支总体形态正常,但局部可见一些分支血管管径不均、局部变细的现象。这些改变提示可能存在血管狭窄或闭塞,但从目前图像质量尚难以定论,建议行进一步检查明确。此外,有1-2支细小分支血管末梢出现中断,但由于分支级数较高,定位比较困难,还需结合更多临床信息判断其意义。\n"
                        "综上,该血管造影图像展示了较为典型的血管解剖,标注质量尚可,但局部血管改变值得关注,建议联系病史、体征做进一步定位和评估,以明确病变性质、程度和范围。"

            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "总体来说你分析的不错,大致学会了分析这些脑血管图像的方法，请注意在后面的分析中不要编造内容,并严格按照格式回答，尽可能使用刚刚提供的关键词 下面看看按照上面的要求分析一下这张图"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "None"
                }
            }
        ]
    }
]


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

        # 记录开始时间的文本
        self.start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        self.auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(self.auth, endpoint, bucket_name)

    def upload_image(self, image_path, folder=None):
        print("Uploading image to Aliyun OSS...")
        file_key = image_path.split('/')[-1]
        if folder is not None:
            if isinstance(folder, str):
                if not folder.endswith("/"):
                    folder += "/"
                file_key = folder + file_key
            elif isinstance(folder, bool):
                if folder:
                    folder = self.start_time
                    file_key = folder + "/" + file_key
            else:
                raise ValueError("Invalid folder parameter. Expected a string or a boolean")
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

    def upload_numpy_array(self, array: np.array, file_name=None, folder=None):
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
        elif ".png" not in file_name:
            file_name = file_name + ".png"
        # 确保数组是二维的
        if array.ndim != 2:
            raise ValueError("Only 2D arrays are supported.")

        if folder is not None:
            if isinstance(folder, str):
                if not folder.endswith("/"):
                    folder += "/"
                file_name = folder + file_name
            elif isinstance(folder, bool):
                if folder:
                    folder = self.start_time
                    file_name = folder + "/" + file_name
            else:
                raise ValueError("Invalid folder parameter. Expected a string or a boolean")


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
    def __init__(self, api_key=None, messages=None):
        if messages is None:
            self.messages = message
        if api_key is None:
            api_key = "11674de81b52a244985b70ad1dc9873f.C6wucI9fuZzgbJuZ"  # "11674de81
        self.client = ZhipuAI(api_key=api_key)
        self.image_host = AliyunOSSImageHost()

    def generate_text(self, image_url):
        messages = self.messages
        # 将messages的第7个元素的image_url替换为image_url
        messages[6]["content"][1]["image_url"]["url"] = image_url
        # while True:
        # print(messages)
        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="glm-4v",  # 填写需要调用的模型名称,
                    messages=messages
                )
                if response.choices[
                    0].message.content == "抱歉，我目前还没有修改图片的能力。如果您有其他请求，欢迎随时向我提问。谢谢！":
                    print("Retrying...")
                    pass
                else:
                    # break
                    return response.choices[0].message.content if response.choices[0].message.content is not None else "Null"
            except Exception as e:
                print(image_url)
                print(e)
                time.sleep(0.1)  # 等待1秒后重试
        print("Failed to generate text, please try a different image or prompt.")
        return "Null"

    def generate_text_from_image(self, image_path):
        image_url = self.image_host.upload_image(image_path, folder=True)
        return self.generate_text(image_url)

    def generate_text_from_numpy_array(self, array: np.array, file_name=None):
        image_url = self.image_host.upload_numpy_array(array, file_name=file_name, folder=True)
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
        print("Generating")
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
