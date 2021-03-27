import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, PATH, train):
        self.img_path = os.path.join(PATH, 'img_data')
        self.label_path = os.path.join(PATH, 'label_data')
        self.img_list = os.listdir(self.img_path)
        self.label_list = os.listdir(self.label_path)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.img_list, self.label_list, test_size = 0.2 , random_state=321)
        print("train X : ",len(self.X_train), "test X : ",len(self.X_test), "train Y : ",len(self.Y_train), "test Y:",len(self.Y_test))
        self.img_transform = transforms.Compose([
                                                  transforms.Resize((224, 224)),
                                                  transforms.ToTensor()
                                                ])
        self.tgt_transform = transforms.Compose([
                                                  transforms.Resize((224, 224), interpolation=Image.NEAREST),
                                                  np.asarray,
                                                  torch.FloatTensor
                                                  ])

        print(train)
        if train == True:
            self.X =self.X_train
            self.Y = self.Y_train
        else:
            self.X = self.X_test
            self.Y = self.Y_test
 
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img_name = self.X[idx]
        lab_name = self.Y[idx]
        png_img =Image.open(os.path.join(self.img_path, img_name))
        annot_img = Image.open(os.path.join(self.label_path, lab_name))
        
        png = self.img_transform(png_img)
        annot = self.tgt_transform(annot_img)
        annot = torch.tensor(annot).long()
        return png, annot   





# PATH ='/home/gpu/Workspace/jm/left_atrial/dataset'

# batch_size =16
# train_dataset = CustomDataset(PATH, train=True)
# test_dataset = CustomDataset(PATH, train=False)

# train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = False, num_workers=0)
# test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers=0)

# print(len(train_dataset), len(test_dataset))