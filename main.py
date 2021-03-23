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

device = torch.device('cuda:0')
net = TRAIL_UNET_IID()
model = net.to(device)

criterion = MS_SSIM(max_val=1, window_size=5)
optimizer = toptim.RAdam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001,
                                           threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=4, verbose=True)
early_stopper = EarlyStopping(patience=15, verbose=True)

losses_train = []
losses_val = []
lrs_train = []
lrs_val = []
num_epochs = 150

for epoch in range(1, num_epochs + 1):
    # train for one epoch
    train_loss, lr_train = train(train_loader, model, criterion, epoch, num_epochs, device)

    losses_train.append(train_loss)
    lrs_train.append(lr_train)

    val_loss, lr_val = validate(val_loader, model, criterion, epoch, num_epochs, device)
    losses_val.append(val_loss)
    lrs_val.append(lr_val)

    scheduler.step(val_loss)
    early_stopper(val_loss, model)
    if early_stopper.early_stop:
        print("Early stopping  !! EPOCH %d/ %d - EPOCH LOSS: %.4f ", epoch, num_epochs, val_loss)
        break

dt_string = datetime.now().strftime("%d%m_%H"+'h'+"%M_")
torch.save(model, (dt_string+'-RADAM-IID.pth'))

test_loss_r = test_real(test_rloader, model, criterion, criterion1, device)
test_loss = test(test_loader, model, criterion, criterion1, device)

