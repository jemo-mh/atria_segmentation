import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt



PATH = '/home/gpu/Workspace/jm/left_atrial/dataset'
img_path = os.path.join(PATH, 'img_data')
lab_path = os.path.join(PATH, 'label_data')
img_list = os.listdir(img_path) 
lab_list = os.listdir(lab_path) 
print("total imgs : {} , total labels :{}".format(len(img_list), len(lab_list)))


X_train, X_test, Y_train, Y_test = train_test_split(img_list, lab_list, test_size=0.2, random_state=321)
print("train X : ",len(X_train), "test X : ",len(X_test), "train Y : ",len(Y_train), "test Y:",len(Y_test))



class TrainDataset(Dataset):
    def __init__(self, ROOT_DIR, X,Y):
        # self.png_root = '/home/gpu/Workspace/jm/dental_seg/dataset/img_data'
        # self.annot_root = '/home/gpu/Workspace/jm/dental_seg/dataset/label_data/'
        self.img_path = os.path.join(ROOT_DIR, 'img_data')
        self.lab_path = os.path.join(ROOT_DIR, 'label_data')
        self.fnamelist= os.listdir(self.img_path)
        self.X=X
        self.Y=Y
        # print(len(self.X))
        # print(len(self.Y))
        # print(len(self.fnamelist))
        
        self.img_transform = transforms.Compose([
                                                  transforms.Resize((224, 224)),
                                                  transforms.ToTensor()
                                                ])
        self.tgt_transform = transforms.Compose([
                                                  transforms.Resize((224, 224), interpolation=Image.NEAREST),
                                                  np.asarray,
                                                  torch.LongTensor
                                                  ])
 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # img_name = self.fnamelist[idx]
        img_name = self.X[idx]
        lab_name = self.Y[idx]
        
        # png_img =Image.open(os.path.join(self.png_root, img_name))
        # annot_img = Image.open(os.path.join(self.annot_root, img_name))
        png_img =Image.open(os.path.join(self.img_path, img_name))
        annot_img = Image.open(os.path.join(self.lab_path, lab_name))
        
        png = self.img_transform(png_img)
        annot = self.tgt_transform(annot_img)
        
        return png,annot
    
batch_size=16
train_dataset = TrainDataset(PATH,X_train, Y_train)
train_loader =  DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle=False, num_workers=0)
test_dataset = TrainDataset(PATH,X_test, Y_test)
test_loader  = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=False, num_workers=0)


# print(test_dataset.img_path)
# for X, Y in (train_loader):
#     '''plot img and label data'''
#     plt.figure()
#     plt.subplot(2,2,1)
#     plt.imshow(X[0,0,:,:], cmap='gray')
#     plt.subplot(2,2,2)
#     plt.imshow(X[1,0,:,:], cmap='gray')
#     plt.subplot(2,2,3)
#     plt.imshow(X[2,0,:,:], cmap='gray')
#     plt.subplot(2,2,4)
#     plt.imshow(X[3,0,:,:], cmap='gray')

#     plt.figure()
#     plt.subplot(2,2,1)
#     plt.imshow(Y[0,:,:], cmap='gray')
#     plt.subplot(2,2,2)
#     plt.imshow(Y[1,:,:], cmap='gray')
#     plt.subplot(2,2,3)
#     plt.imshow(Y[2,:,:], cmap='gray')
#     plt.subplot(2,2,4)
#     plt.imshow(Y[3,:,:], cmap='gray')
#     plt.show()
#     print(X.shape, X.dtype)
#     print(Y.shape)