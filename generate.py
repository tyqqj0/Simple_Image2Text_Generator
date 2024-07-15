# -*- CODING: UTF-8 -*-
# @time 2024/7/9 下午8:14
# @Author tyqqj
# @File generate.py
# @
# @Aim

import numpy as np
import torch
from torch.utils import data
from torch import nn

from render_volume.render import render_numpy
from agent.Agent import ImageTextGenerator

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

import os


def get_mha_names(path: str) -> list:
    # 获取文件夹下所有文件名称
    files = os.listdir(path)
    # 获取所有mha文件名称
    mha_names = []
    for file in files:
        if file.endswith('.mha'):
            # 获取文件的绝对路径
            absolute_path = os.path.abspath(os.path.join(path, file))
            # absolute_path = file
            mha_names.append(absolute_path)
    return mha_names


def get_txt_names(path: str) -> list:
    # 获取文件夹下所有文件名称
    files = os.listdir(path)
    # 获取所有mha文件名称
    mha_names = []
    for file in files:
        if file.endswith('.txt'):
            # 获取文件的绝对路径
            absolute_path = os.path.abspath(os.path.join(path, file))
            # absolute_path = file
            mha_names.append(absolute_path)
    return mha_names


def generate_image_text(files, target_dir, skip_existing=True):
    generator = ImageTextGenerator()
    for i, file in enumerate(files):
        # if i > 3:
        #     break
        # 名称取.../*.mha
        file_name = file.split('\\')[-1]
        # 去掉后缀
        file_name = file_name.split('.')[0]

        # 检查目标文件夹中是否已经存在对应的文本文件
        target_file = os.path.join(target_dir, f"{file_name}.txt")
        if os.path.exists(target_file) and skip_existing:
            print(f"Skipping {file_name} as the text file already exists.")
            continue
        elif os.path.exists(target_file):
            print(f"Overwriting existing file: {file_name}.")

        print(f"Processing: {file_name}")
        # 生成图像
        img = render_numpy(file)
        # 生成文本
        text = generator.generate_text_from_numpy_array(img, file_name)
        x = print(text)

        # 将生成的文本保存到目标文件夹中
        # with open(target_file, 'w') as f:
        with open(target_file, 'w', encoding='utf-8') as f:
            print(f"Writing to {target_file}")
            f.write(text)


def main():
    # 图像文件夹路径
    image_dir = 'D:/Data/brains/train/image_crops'
    # 目标文件夹路径
    target_dir = 'generated_texts'
    qualified_dir = 'qualified_texts'

    # 创建目标文件夹(如果不存在)
    os.makedirs(target_dir, exist_ok=True)

    # 获取数据名称列表, 已经在qualified_dir中的文件不再生成, 使用文件名称对比
    target_files = get_mha_names(path=image_dir) # D:/Data/brains/train/image_crops/Normal002.mha
    qualified_files = get_txt_names(path=qualified_dir) # qualified_texts/Normal002.txt
    # print(qualified_files)
    files = []
    for target_file in target_files:
        target_file_name = target_file.split('\\')[-1]
        target_file_name = target_file_name.split('.')[0]
        # print(target_file_name)
        qualified = False
        for qualified_file in qualified_files:
            # print("\t", qualified_file)
            qualified_file_name = qualified_file.split('\\')[-1]
            qualified_file_name = qualified_file_name.split('.')[0]
            if target_file_name == qualified_file_name:
                qualified = True
                break
        if not qualified:

            files.append(target_file)
    print('files', files)

    # 生成图像文本到目标文件夹
    generate_image_text(files, target_dir, skip_existing=False)


if __name__ == '__main__':
    main()
