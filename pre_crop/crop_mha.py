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


# {'ObjectType': 'Image', 'NDims': 3, 'CompressedData': False, 'BinaryData': True, 'BinaryDataByteOrderMSB': False, 'Offset': array([0., 0., 0.]), 'TransformMatrix': array([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.]]), 'CenterOfRotation': array([0., 0., 0.]), 'AnatomicalOrientation': 'RAI', 'ElementSpacing': array([1., 1., 1.]), 'DimSize': array([ 96, 224, 224]), 'ElementType': <class 'numpy.uint16'>})
def add_additional_metadata(image):
    # 添加额外的元数据
    image_meta = image.meta.copy()
    # ObjectType
    image_meta["ObjectType"] = "Image"
    # NDims
    image_meta["NDims"] = 3
    # CompressedData
    image_meta["CompressedData"] = False
    # BinaryData
    image_meta["BinaryData"] = True
    # BinaryDataByteOrderMSB
    image_meta["BinaryDataByteOrderMSB"] = False
    # CenterOfRotation
    image_meta["CenterOfRotation"] = [0.0, 0.0, 0.0]
    # AnatomicalOrientation
    image_meta["AnatomicalOrientation"] = "RAI"
    # DimSize
    image_meta["DimSize"] = image.shape[-3:]
    # ElementType
    image_meta["ElementType"] = np.uint16
    # ElementSpacing
    image_meta["ElementSpacing"] = [1.0, 1.0, 1.0]
    # Offset
    image_meta["Offset"] = [0.0, 0.0, 0.0]
    # TransformMatrix
    image_meta["TransformMatrix"] = np.eye(3)

    # 写回
    image = monai.data.MetaTensor(image, meta=image_meta)

    print("after:", image.meta)

    return image




def fix_metadata(image, label):
    # 修复元数据, 使得元数据中的空间信息与图像数据匹配
    image_meta = image.meta.copy()
    label_meta = label.meta.copy()

    # 打印原始数据的大小
    # print("Original image shape:", image.shape)
    # print("Original image meta:", image_meta)

    # 更新元数据中的空间信息
    image_meta["spatial_shape"] = [image.shape[-3], image.shape[-2], image.shape[-1]]
    label_meta["spatial_shape"] = [label.shape[-3], label.shape[-2], label.shape[-1]]

    # 更新元数据中的原点信息


    image_meta["affine"] = image_meta["original_affine"]
    label_meta["affine"] = label_meta["original_affine"]

    # 去掉原始元数据中的空间信息
    # image_meta.pop("original_affine", None)
    # label_meta.pop("original_affine", None)
    # image_meta.pop("affine", None)
    # label_meta.pop("affine", None)
    image_meta.pop("crop_center", None)
    label_meta.pop("crop_center", None)


    # 打印修复后的数据大小
    # print("Fixed image shape:", image.shape)

    # print("Fixed image meta:", image_meta)

    # 更新图像和标签的元数据
    image = monai.data.MetaTensor(image, meta=image_meta)
    label = monai.data.MetaTensor(label, meta=label_meta)

    # print(image)

    return image, label

def data_to_np(data: monai.data.MetaTensor)-> monai.data.MetaTensor:
    # 将MetaTensor的数据部分的类型从torch转换为numpy.uint16, 以便保存为mha文件
    data_np = data.detach().cpu().numpy() * 65535
    data_np = data_np.astype(np.uint16)
    data = monai.data.MetaTensor(data_np, meta=data.meta)
    return data




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
            # #
            # print("before:", image_meta)

            # #

            # 修复元数据
            cropped_dict["image"], cropped_dict["label"] = fix_metadata(cropped_dict["image"], cropped_dict["label"])
            cropped_dict["image"] = add_additional_metadata(cropped_dict["image"])

            # 将MetaTensor的数据类型从torch转换为numpy.uint16
            cropped_dict["image"] = data_to_np(cropped_dict["image"])
            cropped_dict["label"] = data_to_np(cropped_dict["label"])


            # 在裁剪后的图像和标签的元数据中添加裁剪编号
            cropped_dict["image"].meta["crop_number"] = j
            cropped_dict["label"].meta["crop_number"] = j

            # 将patch_index设置为crop_number
            cropped_dict["image"].meta["patch_index"] = j
            cropped_dict["label"].meta["patch_index"] = j

            # # 使用CropForegroundd自动裁剪前景区域并更新元数据
            # crop_foreground = CropForegroundd(keys=["image", "label"], source_key="image")
            # cropped_dict = crop_foreground(cropped_dict)

            # image_meta = cropped_dict["image"].meta.copy()
            # label_meta = cropped_dict["label"].meta.copy()
            # #
            # print("after:", image_meta)

            # 保存裁剪后的图像
            image_saver(cropped_dict["image"])

            # 保存裁剪后的标签
            label_saver(cropped_dict["label"])




if __name__ == "__main__":

    # 使用示例
    data_dir = "D:/Data/brains/train/"
    # data_dir = "D:/gkw/data/data_json/vessel.json"
    output_dir = ["D:/Data/brains/train/image_crops", "D:/Data/brains/train/label_crops"]
    num_crops_per_image = 1

    create_dataset(data_dir, num_crops_per_image, output_dir)
