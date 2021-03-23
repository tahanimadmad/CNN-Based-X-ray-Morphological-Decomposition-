# Imports
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

num_epochs = int(sys.argv[1])

# Create pytorch dataset

class TRAILIID(Dataset):

    def __init__(self, root, subset="train"):

        """
        :param root: it has to be a path to the folder that contains the dataset folders
        :param subset: selection of subset folder
        """

        # initialize variables
        self.root = os.path.expanduser(root)
        self.subset = subset
        self.def_path, self.lpw_path = [], []

        def load_images(path):
            images_dir = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
            return images_dir

        self.def_path = load_images(self.root + self.subset + "/def/")
        self.lpw_path = load_images(self.root + self.subset + "/pwc/")
        self.fra_path = load_images(self.root + self.subset + "/fra/")
        self.diff_path = load_images(self.root + self.subset + "/diff/")

    def transform(self, _input, _fra, _lpw):

        _fra1 = _fra.getchannel(0)
        _fra2 = _fra.getchannel(1)

        _input = TTF.to_grayscale(_input, 1)
        _fra1 = TTF.to_grayscale(_fra1, 1)
        _fra2 = TTF.to_grayscale(_fra2, 1)
        _lpw = TTF.to_grayscale(_lpw, 1)

        _input = TTF.resize(_input, (sz, sz), 2)
        _fra1 = TTF.resize(_fra1, (sz, sz), 2)
        _fra2 = TTF.resize(_fra2, (sz, sz), 2)
        _lpw = TTF.resize(_lpw, (sz, sz), 2)

        if random.random() > 0.5:
            _input = TTF.hflip(_input)
            _fra1 = TTF.hflip(_fra1)
            _fra2 = TTF.hflip(_fra2)
            _lpw = TTF.hflip(_lpw)

        if random.random() > 0.5:
            _input = TTF.vflip(_input)
            _fra1 = TTF.vflip(_fra1)
            _fra2 = TTF.vflip(_fra2)
            _lpw = TTF.vflip(_lpw)

        if random.random() < 0.5:
            contrast_factor = 0.5
            _input = TTF.adjust_contrast(_input, contrast_factor)
            _fra1 = TTF.adjust_contrast(_fra1, contrast_factor)
            _fra2 = TTF.adjust_contrast(_fra2, contrast_factor)
            _lpw = TTF.adjust_contrast(_lpw, contrast_factor)

        if random.random() < 0.5:
            brightness_factor = 0.5
            _input = TTF.adjust_brightness(_input, brightness_factor)
            _fra1 = TTF.adjust_brightness(_fra1, brightness_factor)
            _fra2 = TTF.adjust_brightness(_fra2, brightness_factor)
            _lpw = TTF.adjust_brightness(_lpw, brightness_factor)

        _input = TTF.to_tensor(_input)
        _fra1 = TTF.to_tensor(_fra1)
        _fra2 = TTF.to_tensor(_fra2)
        _lpw = TTF.to_tensor(_lpw)

        # Normalization

        _input -= torch.min(_input)
        if torch.max(_input) != 0:
            _input /= torch.max(_input)
        _fra1 -= torch.min(_fra1)
        if torch.max(_fra1) != 0:
            _fra1 /= torch.max(_fra1)
        _fra2 -= torch.min(_fra2)
        if torch.max(_fra2) != 0:
            _fra2 /= torch.max(_fra2)
        _lpw -= torch.min(_lpw)
        if torch.max(_lpw) != 0:
            _lpw /= torch.max(_lpw)

        return _input, _fra1, _fra2, _lpw

    def __getitem__(self, index):

        """
        :param index: image index
        :return: tuple (_def = img, _lpw = target) with the input data and its ground truth
        """

        _def = Image.open(self.def_path[index])
        _lpw = Image.open(self.lpw_path[index])
        _diff = Image.open(self.diff_path[index])
        _def, _diff1, _diff2, _lpw = self.transform(_def, _diff, _lpw)

        return _def, _diff1, _diff2, _lpw, index

    def __len__(self):
        return len(self.def_path)
