# -*- CODING: UTF-8 -*-
# @time 2024/7/9 下午8:31
# @Author tyqqj
# @File manual_selection.py
# @
# @Aim 

import numpy as np
import torch
from torch.utils import data
from torch import nn

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

import os
import shutil

import keyboard

# 根据操作系统选择适当的清屏命令
if os.name == 'nt':  # Windows
    clear_command = 'cls'
else:  # Unix/Linux/macOS
    clear_command = 'clear'


def read_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return content


def move_file(src_path, dst_path):
    shutil.move(src_path, dst_path)


def manual_selection(temp_dir, target_dir):
    # 获取预存文件夹中的所有文件
    files = os.listdir(temp_dir)

    for file in files:
        file_path = os.path.join(temp_dir, file)

        # 读取文件内容
        content = read_file(file_path)

        while True:

            os.system(clear_command)  # 清屏
            print(f"File: {file}")
            print("Content:")
            print(content)
            print("=" * 50)

            print("Move this file to the target directory?")
            print("Press 'y' for Yes, 'n' for No, or any other key to skip.")
            choice = keyboard.read_key().lower()

            if choice == 'y':
                # 移动文件到目标文件夹
                dst_path = os.path.join(target_dir, file)
                move_file(file_path, dst_path)
                print(f"Moved {file} to the target directory.")
                break
            elif choice == 'n':
                print(f"Skipped {file}.")
                break
            else:
                print("Invalid input. Please try again.")
                # break

        print("-" * 50)
    input("Press Enter to continue...")


def main():
    # 预存文件夹路径
    temp_dir = 'generated_texts'
    # 目标文件夹路径
    target_dir = 'qualified_texts'

    # 创建目标文件夹(如果不存在)
    os.makedirs(target_dir, exist_ok=True)

    # 人工筛选文件并移动到目标文件夹
    manual_selection(temp_dir, target_dir)


if __name__ == '__main__':
    main()
