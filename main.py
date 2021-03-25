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

# from Dataset import train_loader, test_loader
# from UNet import Build_UNet, Conv_Block, Upconv_Block

# import data_loader
# import UNet_1
import UNet_3


# import UNet
import Dataset
# print(Dataset.X_train)
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

PATH ='/home/gpu/Workspace/jm/left_atrial/dataset'
train_dataset = Dataset.train_dataset
test_dataset = Dataset.test_dataset

train_loader = Dataset.train_loader
test_loader= Dataset.test_loader

# unet = UNet_1.Build_UNet()
unet = UNet_3.UNet().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(unet.parameters(), lr = 0.001)

epochs = 100
for epoch in range(epochs):
    avg_loss=0
    avg_acc =0
    total_batch = len(train_dataset)//Dataset.batch_size
    # print(total_batch)

    for i, (batch_img, batch_lab) in enumerate(train_loader):
        X =batch_img.to(device)
        Y = batch_lab.to(device)
        print("X.shape", X.shape)
        print(X.dtype)
        print("Y.shape", Y.shape)
        print(Y.dtype)        
        optimizer.zero_grad()
        y_pred = unet.forward(X)
        print(y_pred.shape, y_pred.dtype)
        _,predicted = torch.max(y_pred.data, 1)
        print(predicted.shape, predicted.dtype)
        loss = criterion(predicted.to(torch.float), Y.to(torch.float))

        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        
        if (i+1)%10 ==0:
            print("Epoch: ", epoch+1, "iteration : ", i+1, "loss: ", loss.item())


# X= torch.randn(16, 1,256,256).to(device)
# print(X.shape)
# y_pred = unet(X)
# # print(y_pred)


# m = nn.Sigmoid()
# loss = nn.BCELoss()
# input = torch.randn(3, requires_grad=True)
# target = torch.empty(3).random_(2)
# output = loss(m(input), target)
# print(target.dtype)
# output.backward()