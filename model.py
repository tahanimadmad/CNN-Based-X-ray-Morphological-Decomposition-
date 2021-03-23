import os
from os.path import isfile, join
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil
from random import randint
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from tqdm import tqdm
import random
from torchvision.utils import save_image
import sys
import torchvision.transforms.functional as TTF
from math import exp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from datetime import datetime
import torch_optimizer as toptim


# Model

# Create the U-net model

# Convolutional block
class conv_block(nn.Module):

    def __init__(self, in_ch, out_ch, ker, pad):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=ker, stride=1, padding=pad, bias=True),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, kernel_size=ker, stride=1, padding=pad, bias=True),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


# Up block
class up_block(nn.Module):

    def __init__(self, in_ch, out_ch, ker, pad):
        super(up_block, self).__init__()

        self.up = nn.Sequential(nn.Upsample(scale_factor=2),
                                nn.Conv2d(in_ch, out_ch, kernel_size=ker, stride=1, padding=pad, bias=True),
                                )

    def forward(self, x):
        x = self.up(x)
        return x

    # U-net network definition


class TRAIL_UNET_IID(nn.Module):
    """
    Define UNET model that accepts a 128 input and mostly uses 3x3 kernels
    with stride and padding = 1. It reduces the size of the image to XXXXXXX pixels
    ** It might not work if the input 'x' is not a square.
    """

    def __init__(self):
        super(TRAIL_UNET_IID, self).__init__()

        self.down_0 = conv_block(1, 32, 3, 1)

        self.down_1 = conv_block(32, 64, 3, 1)

        self.down_2 = conv_block(64, 128, 3, 1)

        self.down_3 = conv_block(128, 256, 3, 1)

        self.down_4 = conv_block(256, 512, 3, 1)

        self.middle = conv_block(512, 1024, 1, 0)

        self.up_4 = conv_block(512, 512, 3, 1)
        self.arrow4 = up_block(1024, 512, 3, 1)

        self.up_3 = conv_block(256, 256, 3, 1)
        self.arrow3 = up_block(512, 256, 3, 1)

        self.up_2 = conv_block(128, 128, 3, 1)
        self.arrow2 = up_block(256, 128, 3, 1)

        self.up_1 = conv_block(64, 64, 3, 1)
        self.arrow1 = up_block(128, 64, 3, 1)

        self.up_0 = conv_block(32, 32, 3, 1)
        self.arrow0 = up_block(64, 32, 3, 1)

        self.output_fra = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.output_lpw = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # LPW

        down_lpw_0 = self.down_0(x)
        out_lpw_0 = F.max_pool2d(down_lpw_0, kernel_size=2, stride=2)

        down_lpw_1 = self.down_1(out_lpw_0)
        out_lpw_1 = F.max_pool2d(down_lpw_1, kernel_size=2, stride=2)

        down_lpw_2 = self.down_2(out_lpw_1)
        out_lpw_2 = F.max_pool2d(down_lpw_2, kernel_size=2, stride=2)

        down_lpw_3 = self.down_3(out_lpw_2)
        out_lpw_3 = F.max_pool2d(down_lpw_3, kernel_size=2, stride=2)

        down_lpw_4 = self.down_4(out_lpw_3)
        out_lpw_4 = F.max_pool2d(down_lpw_4, kernel_size=2, stride=2)

        out_lpw_5 = self.middle(out_lpw_4)

        out_lpw_6 = self.arrow4(out_lpw_5)
        out_lpw_6 = out_lpw_6.add(down_lpw_4)
        out_lpw_6 = self.up_4(out_lpw_6)

        out_lpw_7 = self.arrow3(out_lpw_6)
        out_lpw_7 = out_lpw_7.add(down_lpw_3)
        out_lpw_7 = self.up_3(out_lpw_7)

        out_lpw_8 = self.arrow2(out_lpw_7)
        out_lpw_8 = out_lpw_8.add(down_lpw_2)
        out_lpw_8 = self.up_2(out_lpw_8)

        out_lpw_9 = self.arrow1(out_lpw_8)
        out_lpw_9 = out_lpw_9.add(down_lpw_1)
        out_lpw_9 = self.up_1(out_lpw_9)

        out_lpw_10 = self.arrow0(out_lpw_9)
        out_lpw_10 = out_lpw_10.add(down_lpw_0)
        out_lpw_10 = self.up_0(out_lpw_10)

        # FRA

        down_fra_0 = self.down_0(x)
        out_fra_0 = F.max_pool2d(down_fra_0, kernel_size=2, stride=2)

        down_fra_1 = self.down_1(out_fra_0)
        out_fra_1 = F.max_pool2d(down_fra_1, kernel_size=2, stride=2)

        down_fra_2 = self.down_2(out_fra_1)
        out_fra_2 = F.max_pool2d(down_fra_2, kernel_size=2, stride=2)

        down_fra_3 = self.down_3(out_fra_2)
        out_fra_3 = F.max_pool2d(down_fra_3, kernel_size=2, stride=2)

        down_fra_4 = self.down_4(out_fra_3)
        out_fra_4 = F.max_pool2d(down_fra_4, kernel_size=2, stride=2)

        out_fra_5 = self.middle(out_fra_4)

        out_fra_6 = self.arrow4(out_fra_5)
        out_fra_6 = out_fra_6.add(down_fra_4)
        out_fra_6 = self.up_4(out_fra_6)

        out_fra_7 = self.arrow3(out_fra_6)
        out_fra_7 = out_fra_7.add(down_fra_3)
        out_fra_7 = self.up_3(out_fra_7)

        out_fra_8 = self.arrow2(out_fra_7)
        out_fra_8 = out_fra_8.add(down_fra_2)
        out_fra_8 = self.up_2(out_fra_8)

        out_fra_9 = self.arrow1(out_fra_8)
        out_fra_9 = out_fra_9.add(down_fra_1)
        out_fra_9 = self.up_1(out_fra_9)

        out_fra_10 = self.arrow0(out_fra_9)
        out_fra_10 = out_fra_10.add(down_fra_0)
        out_fra_10 = self.up_0(out_fra_10)

        return self.output_lpw(out_lpw_10), self.output_fra(out_fra_10)
        
        

