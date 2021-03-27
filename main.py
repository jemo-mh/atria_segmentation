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
from UNet_1 import Build_UNet

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
PATH ='/home/gpu/Workspace/jm/left_atrial/dataset'

batch_size =16
train_dataset = CustomDataset(PATH, train=True)
test_dataset = CustomDataset(PATH, train=False)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = False, num_workers=0)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers=0)

print(len(train_dataset), len(test_dataset))

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss()

    def forward(self, inputs, targets):
        return self.nll_loss(nn.functional.log_softmax(inputs, dim=1), targets)



# unet = UNet().to(device)
unet= Build_UNet(input_channel=1, num_classes=2).to(device)
# criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()
criterion = CrossEntropyLoss2d()
m= nn.Sigmoid()
optimizer = optim.Adam(unet.parameters(), lr = 0.001)

epochs =50

# for epoch in range(epochs):
#     avg_loss=0
#     avg_acc =0
#     total_batch = len(train_dataset)//batch_size
#     print(total_batch)

#     for i, (batch_img, batch_lab) in enumerate(train_loader):
#         X =batch_img.to(device)
#         Y = batch_lab.to(device, dtype = torch.long)
#         print("X.shape", X.shape)
#         print(X.dtype)
#         print("Y.shape", Y.shape)
#         print(Y.dtype)        
#         optimizer.zero_grad()
#         y_pred = unet.forward(X)
#         print("ypred ", y_pred.shape, y_pred.dtype)
#         # _,predicted = torch.max(y_pred.data, 1)
        
#         # print("predicted",predicted, predicted.shape, predicted.dtype)
#         # loss = criterion(m(predicted), Y.type(torch.float))
#         # loss = criterion(predicted, Y)
#         loss = criterion(y_pred, Y)

#         loss.backward()
#         optimizer.step()
#         avg_loss += loss.item()
    
#         if (i+1)%10 ==0:
#             print("Epoch: ", epoch+1, "iteration : ", i+1, "loss: ", loss.item())

for epoch in range(epochs):
    avg_loss = 0
    avg_acc = 0
    total_batch = len(train_dataset) // batch_size
    for i, (batch_img, batch_lab) in enumerate(train_loader):
        X = batch_img.to(device)
        Y = batch_lab.to(device)

        optimizer.zero_grad()

        y_pred = unet.forward(X)
        # print(y_pred)
        # y_pred = y_pred.long()

        loss = criterion(y_pred, Y)
        
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()

        if (i+1)%20 == 0 :
            print("Epoch : ", epoch+1, "Iteration : ", i+1, " Loss : ", loss.item())