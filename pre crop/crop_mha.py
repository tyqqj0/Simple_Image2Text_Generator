# -*- CODING: UTF-8 -*-
# @time 2024/6/27 下午12:39
# @Author tyqqj
# @File crop_mha.py
# @
# @Aim
import json

import numpy as np
import torch
from torch.utils import data
from torch import nn

import monai

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

import os
import numpy as np
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    ToTensord,
    SaveImaged, Compose, SaveImage,
)
from monai.data import Dataset


def custom_name_formatter(metadict: dict, saver: monai.transforms.Transform) -> dict:
    """自定义命名格式化函数"""
    print("Using custom name formatter...")
    subject = metadict.get(monai.utils.ImageMetaKey.FILENAME_OR_OBJ, getattr(saver, "_data_index", 0))
    patch_index = metadict.get(monai.utils.ImageMetaKey.PATCH_INDEX, None)
    crop_number = metadict.get("crop_number", None)  # 获取裁剪编号
    if crop_number is not None:
        return {"subject": f"{subject}_crop_{crop_number}", "idx": patch_index}
    else:
        return {"subject": f"{subject}", "idx": patch_index}


def load_data_dicts(data_path):
    if isinstance(data_path, str):
        if data_path.endswith(".json"):
            # 如果传入的是 JSON 文件路径,读取 JSON 文件作为数据字典
            with open(data_path, "r") as f:
                data_dicts = json.load(f)
                if 'train' in data_dicts.keys():
                    data_dicts = data_dicts['train']
        else:
            # 如果传入的是文件夹路径,自动搜索所有的图像和标签文件
            data_path_image = os.path.join(data_path, "image")
            data_path_label = os.path.join(data_path, "label")
            image_files = sorted([os.path.join(data_path_image, f).replace("\\", "/") for f in os.listdir(data_path_image) if f.endswith(".mha")])
            label_files = sorted([os.path.join(data_path_label, f).replace("\\", "/") for f in os.listdir(data_path_label) if f.endswith(".mha")])

            data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(image_files, label_files)]
    elif isinstance(data_path, list):
        # 如果传入的是数据字典列表,直接使用
        data_dicts = data_path
    else:
        raise ValueError("Unsupported data path format. Expected a string or a list of dictionaries.")

    return data_dicts


def create_dataset(data_dir, num_crops_per_image, output_dirs, max_num=-1):
    # 定义transforms
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ])

    # 加载数据集
    # data_dicts = [
    #     {"image": os.path.join(data_dir, "Normal002.mha"), "label": os.path.join(data_dir.replace("image", "label"), "Normal002.mha")},
    # ]
    data_dicts = load_data_dicts(data_dir)
    print(data_dicts)
    dataset = Dataset(data=data_dicts, transform=transforms)

    # 对每个图像进行随机裁剪和保存
    for i in range(len(dataset)):
        if max_num > 0 and i >= max_num:
            break
        image_dict = dataset[i]
        image_name = os.path.basename(data_dicts[i]["image"]).split(".")[0]  # 提取图像名称(不包括扩展名)
        print(f"Processing {image_name}...")

        # 创建 image_saver 和 label_saver,使用自定义命名格式化函数
        image_output_dir = output_dirs[0]
        label_output_dir = output_dirs[1]
        image_saver = SaveImage(output_dir=image_output_dir, output_ext=".mha", separate_folder=False, output_name_formatter=custom_name_formatter,
                                output_postfix="image")
        label_saver = SaveImage(output_dir=label_output_dir, output_ext=".mha", separate_folder=False, output_name_formatter=custom_name_formatter,
                                output_postfix="label")

        for j in range(num_crops_per_image):
            print(f"Processing crop {j}...")
            # 随机裁剪
            cropper = RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            )
            cropped_dict = cropper(image_dict)

            cropped_dict = cropped_dict[0]

            # # 仅保留裁剪区域的有用数据
            # cropped_image = cropped_dict["image"].detach().cpu().numpy()
            # cropped_label = cropped_dict["label"].detach().cpu().numpy()嗯
            #
            # # 复制原始元数据中的信息
            # image_meta = cropped_dict["image"].meta.copy()
            # label_meta = cropped_dict["label"].meta.copy()
            #
            # print(image_meta)
            # #
            # # 更新裁剪后的图像和标签的元数据中的空间信息
            # affine = np.eye(4)
            # affine[:3, :3] = np.diag([1.0, 1.0, 1.0])  # 假设体素大小为 1x1x1
            # cropped_dict["image"] = monai.data.MetaTensor(cropped_image, affine=affine, meta=image_meta)
            # cropped_dict["label"] = monai.data.MetaTensor(cropped_label, affine=affine, meta=label_meta)

            # 在裁剪后的图像和标签的元数据中添加裁剪编号
            cropped_dict["image"].meta["crop_number"] = j
            cropped_dict["label"].meta["crop_number"] = j

            # 将patch_index设置为crop_number
            cropped_dict["image"].meta["patch_index"] = j
            cropped_dict["label"].meta["patch_index"] = j

            # 保存裁剪后的图像
            image_saver(cropped_dict["image"])

            # 保存裁剪后的标签
            label_saver(cropped_dict["label"])


# 使用示例
# data_dir = "D:/Data/brains/train/"
data_dir = "D:/gkw/data/data_json/vessel.json"
output_dir = ["D:/Data/brains/train/image_crops", "D:/Data/brains/train/label_crops"]
num_crops_per_image = 1

create_dataset(data_dir, num_crops_per_image, output_dir)
