# -*- CODING: UTF-8 -*-
# @time 2024/6/26 下午8:56
# @Author tyqqj
# @File main.py
# @
# @Aim
import os

from render import render_numpy
from agent.Agent import ImageTextGenerator


# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision


def get_mha_names(path: str) -> list:
    # 获取文件夹下所有文件名称
    import os
    files = os.listdir(path)
    # 获取所有mha文件名称
    mha_names = []
    for file in files:
        if file.endswith('.mha'):
            mha_names.append(file)
    return mha_names


def generate_image_text(files):
    generator = ImageTextGenerator()
    texts = {}
    for file in files:
        print("Processing: ", file)
        # 生成图像
        img = render_numpy(file)
        # 生成文本
        text = generator.generate_text_from_numpy_array(img)
        print(text)
        # 名称取.../*.mha
        file = file.split('/')[-1]
        # 去掉后缀
        file = file.split('.')[0]
        texts[file] = text
    return texts

    # 生成文本


def save_texts(texts, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for k, v in texts.items():
        with open(f'{path}/{k}.txt', 'w') as f:
            f.write(v)


def main():
    # 获取数据名称列表
    files = get_mha_names(path='D:/Data/brains/train/image')
    print(files)

    # 生成图像文本
    texts = generate_image_text(files)

    # 保存文本
    with open('agent/texts.txt', 'w') as f:
        for k, v in texts.items():
            f.write(f'{k}: {v}\n')

    # 保存文本
    save_texts(texts, 'texts')


if __name__ == '__main__':
    main()
