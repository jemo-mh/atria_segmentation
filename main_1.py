#lib
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, utils
import os
from PIL import Image
import torch
from tqdm import tqdm
from torch import nn, optim
#.py
from utils.Dataset import CustomDataset
# from UNet_3 import UNet
from UNet_1 import Build_UNet

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
PATH ='/home/gpu/Workspace/jm/left_atrial/dataset'

batch_size =16
train_dataset = CustomDataset(PATH, train=True)
test_dataset = CustomDataset(PATH, train=False)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = False, num_workers=0)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers=0)

print(len(train_dataset), len(test_dataset))


# unet = UNet().to(device)
unet =Build_UNet().to(device)
criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()

m= nn.Sigmoid()
optimizer = optim.Adam(unet.parameters(), lr = 0.001)

epochs =50

for X, Y in train_loader:
    # print("X", X)
    # print("Y",Y)
    X = np.transpose(X,[0,2,3,1])
    X = X.to(device)
    Y = Y.to(device)
    y_pred = unet(X).forward
    print(y_pred.dtype, y_pred.shape)