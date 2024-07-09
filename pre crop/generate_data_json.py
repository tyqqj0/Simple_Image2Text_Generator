# -*- CODING: UTF-8 -*-
# @time 2023/10/20 16:07
# @Author tyqqj
# @File generate_data_json.py
# @
# @Aim


import json
import os


def get_a_set(data_dir):
    # 图像和标签的目录
    images_dir = data_dir["image"]
    labels_dir = data_dir["label"]

    print(images_dir, labels_dir)
    # 获取所有图像和标签文件的路径
    image_files = sorted(os.listdir(images_dir))
    label_files = sorted(os.listdir(labels_dir))
    # 创建数据列表
    datalist = [
        {"image": os.path.join(images_dir, img).replace('\\', '/'),
         "label": os.path.join(labels_dir, lbl).replace('\\', '/')}
        for img, lbl in zip(image_files, label_files)
    ]

    if "max_amount" in data_dir.keys():
        max_amount = data_dir["max_amount"]
        datalist = datalist[:max_amount]

    if "shuffle_rate" in data_dir.keys():
        shuffle_rate = data_dir["shuffle_rate"]
        import random
        random.seed(6)
        # 随机打乱一定比例的标签

        # 计算需要打乱的元素数量
        shuffle_amount = int(len(datalist) * shuffle_rate)

        # 获取需要打乱的元素
        shuffle_part = datalist[:shuffle_amount]

        # 随机选择需要打乱的元素
        shuffle_indices = random.sample(range(len(datalist)), shuffle_amount)

        # 提取需要打乱的标签
        labels_to_shuffle = [datalist[i]['label'] for i in shuffle_indices]

        # 打乱标签
        random.shuffle(labels_to_shuffle)

        # 将打乱后的标签重新赋值给数据列表的对应元素
        for i, label in zip(shuffle_indices, labels_to_shuffle):
            datalist[i]['label'] = label

    return datalist


def get_dsets(dirstt):
    all_lists = {}
    for key, value in dirstt.items():
        all_lists[key] = get_a_set(value)
    print(json.dumps(all_lists, indent=4))
    return all_lists


def save_json(all_lists, target_file, target_dir=None):
    if target_dir is None:
        target_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, target_file)
    with open(target_path, "w") as dlj:
        json.dump(all_lists, dlj, indent=4)
    print(f"JSON file saved at: {target_path}")


if __name__ == "__main__":
    root = "D:\\Data\\brains"
    dirstt = {
        "train": {
            "image": root + "\\train\\image_crops",
            "label": root + "\\train\\label_crops",
            # "shuffle_rate": 0.40
        },
        "val": {
            "image": root + "\\train\\image_crops",
            "label": root + "\\train\\label_crops",
            "max_amount": 6
        }
    }
    all_lists = get_dsets(dirstt)
    target_file = "vessel_pre_cropped.json"
    target_dir = "../data"
    save_json(all_lists, target_file, target_dir)
# get_a_set()
