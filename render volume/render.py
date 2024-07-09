# -*- CODING: UTF-8 -*-
# @time 2024/6/26 下午8:06
# @Author tyqqj
# @File render.py
# @
# @Aim 

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torch import nn
from vanilla_roll.anatomy_orientation import (
    CSA,  # 体积
    AnatomyAxis,  # 解剖轴
    AnatomyOrientation,  # 解剖方向
    Axial,  # 在轴向上
    Coronal,  # 在冠状面上
    Sagittal,  # 在矢状面上
    get_direction,  # 获取方向
)

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import skimage.io

import vanilla_roll as vr

# from A high-resolution 7-Tesla fMRI dataset from complex natural stimulation with an audio movie
# https://www.openfmri.org/dataset/ds000113/
MRA_FILE_URL = "https://s3.amazonaws.com/openneuro/ds000113/ds000113_unrevisioned/uncompressed/sub003/angio/angio001.nii.gz"  # noqa: E501


def fetch_mra_volume() -> vr.volume.Volume:
    with TemporaryDirectory() as tmpdir:
        mra_file = Path(tmpdir) / "mra.nii.gz"
        urllib.request.urlretrieve(MRA_FILE_URL, mra_file)
        return vr.io.read_nifti(mra_file)


def get_mha_volume() -> vr.volume.Volume:
    path = "D:/Data/brains/train/image/Normal002.mha"
    return vr.io.read_mha(path)


def save_result(ret: vr.rendering.types.RenderingResult, path: str):
    img_array = vr.rendering.convert_image_to_array(ret.image)

    # 将图像数组转换为 PIL 图像对象
    img = Image.fromarray(np.from_dlpack(img_array))

    # 将图像转换为 "L" 模式 (8 位灰度)
    img = img.convert("L")

    # 保存图像为 PNG 格式
    img.save(path, format="PNG")
    # skimage.io.imsave(path, np.from_dlpack(img_array))  # type: ignore


def get_result(ret: vr.rendering.types.RenderingResult):
    img_array = vr.rendering.convert_image_to_array(ret.image)

    # 将图像数组转换为 PIL 图像对象
    img = Image.fromarray(np.from_dlpack(img_array))

    # 将图像转换为 "L" 模式 (8 位灰度)
    img = img.convert("L")

    return img


def get_numpy_array(ret: vr.rendering.types.RenderingResult):
    img_array = vr.rendering.convert_image_to_array(ret.image)
    return np.from_dlpack(img_array)


def render_numpy(path: str):
    volume = get_mha_volume()
    ret = vr.render(volume, mode=vr.rendering.mode.MIP(), face=Coronal.RIGHT, spacing=1.0)
    return get_numpy_array(ret)


def main():
    volume = get_mha_volume()
    ret = vr.render(volume, mode=vr.rendering.mode.MIP(), face=Coronal.RIGHT, spacing=1.0)
    save_result(ret, f"result.png")


if __name__ == "__main__":
    main()
