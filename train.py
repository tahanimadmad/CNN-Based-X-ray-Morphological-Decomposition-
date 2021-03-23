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

# Define the training function
def train(train_loader, model, criterion, epoch, num_epochs, device):
    LRs = AverageMeter()
    losses = AverageMeter()
    # Progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels_fra1, labels_fra2, labels_lpw, index) in pbar:
        # Convert torch tensor to Variable
        images = images.to(device, dtype=torch.float)
        labels_fra1 = labels_fra1.to(device, dtype=torch.float)
        labels_fra2 = labels_fra2.to(device, dtype=torch.float)
        labels_lpw = labels_lpw.to(device, dtype=torch.float)

        # Compute output
        optimizer.zero_grad()
        output_lpw, output_fra = model(images)
        output_fra1 = output_fra[:, 0, :, :].unsqueeze(1)
        output_fra2 = output_fra[:, 1, :, :].unsqueeze(1)

        # Measure loss
        loss_fra1 = 1 - criterion(output_fra1, labels_fra1)
        loss_fra2 = 1 - criterion(output_fra2, labels_fra2)
        loss_lpw = 1 - criterion(output_lpw, labels_lpw)

        # Total losses
        recon = output_lpw - (1 - output_fra1) + (1 - output_fra2)
        loss_rec = 1 - criterion(recon, images)

        if np.isnan(loss_fra1.item()) or np.isnan(loss_fra2.item()) or np.isnan(loss_lpw.item()) or np.isnan(
                loss_rec.item()):
            print(index)
            sys.exit("nan")

        loss_tot = (loss_fra1 + loss_fra2 + 2 * loss_lpw + loss_rec) / 5
        loss_total = loss_tot.item()
        losses.update(loss_total, images.size(0))

        # compute gradient and do SGD step
        loss_tot.backward()
        optimizer.step()

        # update progress bar status
        LR = get_lr(optimizer)
        LRs.update(LR, images.size(0))
        pbar.set_description('[TRAIN] - EPOCH %d/ %d - BATCH LOSS: %.4f/ %.4f(avg)  - [Lr]: %.4f '
                             % (epoch, num_epochs, losses.val, losses.avg, LR))

    # return avg loss over the epoch
    return losses.avg, LRs.avg


# define the training function
def validate(val_loader, model, criterion, epoch, num_epochs, device):
    LRs = AverageMeter()
    losses = AverageMeter()
    # Progress bar
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for i, (images, labels_fra1, labels_fra2, labels_lpw, index) in pbar:
        # Convert torch tensor to Variable
        images = images.to(device, dtype=torch.float)
        labels_fra1 = labels_fra1.to(device, dtype=torch.float)
        labels_fra2 = labels_fra2.to(device, dtype=torch.float)
        labels_lpw = labels_lpw.to(device, dtype=torch.float)

        # compute output
        optimizer.zero_grad()
        output_lpw, output_fra = model(images)
        output_fra1 = output_fra[:, 0, :, :].unsqueeze(1)
        output_fra2 = output_fra[:, 1, :, :].unsqueeze(1)

        # measure loss
        loss_fra1 = 1 - criterion(output_fra1, labels_fra1)
        loss_fra2 = 1 - criterion(output_fra2, labels_fra2)
        loss_lpw = 1 - criterion(output_lpw, labels_lpw)

        # Total losses
        recon = output_lpw - (1 - output_fra1) + (1 - output_fra2)
        loss_rec = 1 - criterion(recon, images)

        if np.isnan(loss_fra1.item()) or np.isnan(loss_fra2.item()) or np.isnan(loss_lpw.item()) or np.isnan(
                loss_rec.item()):
            print(index)
            sys.exit("nan")

        loss_tot = (loss_fra1 + loss_fra2 + 2 * loss_lpw + loss_rec) / 5
        loss_total = loss_tot.item()
        losses.update(loss_total, images.size(0))

        # update progress bar status
        LR = get_lr(optimizer)
        LRs.update(LR, images.size(0))
        pbar.set_description('[VALIDATION] - EPOCH %d/ %d - BATCH LOSS: %.4f/ %.4f(avg)  - [Lr]: %.4f '
                             % (epoch, num_epochs, losses.val, losses.avg, LR))

    # return avg loss over the epoch
    return losses.avg, LRs.avg

