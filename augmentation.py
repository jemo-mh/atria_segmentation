import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
img_path = '/home/gpu/Workspace/jm/left_atrial/dataset/img_data'
label_path = '/home/gpu/Workspace/jm/left_atrial/dataset/label_data'
img_list = os.listdir(img_path)
label_list = os.listdir(label_path)


print("number of original data",len(img_list), len(label_list))
count=0
count1=0
count2=0
for i in range(len(img_list)):
    image = Image.open(os.path.join(label_path,label_list[i]))
    image= np.array(image)
    non_zero = np.count_nonzero(image)
    if non_zero ==0 :
        # plt.imshow(image, cmap='gray')
        # plt.show()
        # print(image)
        count+=1
        dl_img_path = os.path.join(img_path, str(i)+'.png')
        dl_label_path = os.path.join(label_path, str(i)+'.png')
        if os.path.isfile(dl_img_path):
            os.remove(dl_img_path)
            count1+=1
        if os.path.isfile(dl_label_path):
            os.remove(dl_label_path)
            count2+=1


        # print(dl_img_path, dl_label_path)

print(count)
print(count1, count2)

img_path = '/home/gpu/Workspace/jm/left_atrial/dataset/img_data'
label_path = '/home/gpu/Workspace/jm/left_atrial/dataset/label_data'
img_list = os.listdir(img_path)
label_list = os.listdir(label_path)


print("number after deleting empty label data",len(img_list), len(label_list))
count=0
# path = '/home/gpu/Workspace/jm/left_atrial/dataset/label_data/3.png'
# image = Image.open(path)
# image= np.array(image)
# plt.imshow(image, cmap='gray')
# plt.show()
