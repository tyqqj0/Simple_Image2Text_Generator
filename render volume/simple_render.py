# -*- CODING: UTF-8 -*-
# @time 2024/6/25 下午7:13
# @Author tyqqj
# @File simple_render.py
# @
# @Aim 

import numpy as np
import torch
import vtk
from matplotlib import pyplot as plt
from torch.utils import data
from torch import nn

from mha_loader import mhaReader


# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision



if __name__ == '__main__':
    test_path = 'D:/Data/brains/train/image/Normal002.mha'
    reader = mhaReader()
    arr, file_name = reader(test_path)
    # arr = vtkTonp(arr)
    arr = torch.tensor(arr)
    print(arr.shape)
    print(file_name)
    mean_x = dimention_mean(arr, dim=0)
    plt.imshow(mean_x)
    plt.show()
