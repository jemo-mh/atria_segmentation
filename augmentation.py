import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
import imageio
from tqdm import tqdm

img_path = '/home/gpu/Workspace/jm/left_atrial/dataset/img_data'
label_path = '/home/gpu/Workspace/jm/left_atrial/dataset/label_data'
img_list = os.listdir(img_path)
label_list = os.listdir(label_path)

# dl_img =[]
# for i in label_list:
#     image = Image.open(os.path.join(label_path, i))
#     # print(image)
#     zero = np.count_nonzero(image)
#     if zero ==0:
#         # print(i, zero)
#         dl_img.append(i)

# print("total img : ",len(img_list), "empty label : ", len(dl_img), "img to use : ",len(img_list)-len(dl_img))

# count=0
# for i in dl_img:
#     dl_img_path = os.path.join(img_path, i)
#     dl_label_path = os.path.join(label_path, i)
#     if os.path.isfile(dl_img_path):
#         os.remove(dl_img_path)
#     if os.path.isfile(dl_label_path):
#         os.remove(dl_label_path)

ia.seed(4)
for i in tqdm(range(len(img_list))):
    image = imageio.imread(os.path.join(img_path, img_list[i]))
    label = imageio.imread(os.path.join(label_path, label_list[i]))
    # ia.imshow(image)
    # print(image.shape, image.dtype)

    rotate1 = iaa.Affine(rotate=(-5,-1))
    rotate2 = iaa.Affine(rotate=(1,5))
    img_aug1 = rotate2(image = image)
    img_aug2 = rotate1(image=image)
    label_aug1 = rotate1(image = label)
    label_aug2 = rotate2(image= label)
    name = img_list[i].split('.')
    name = name[0]
    print(name)
    imageio.imwrite(uri =os.path.join(img_path, name+'_1.png'),im= img_aug1)
    imageio.imwrite(uri =os.path.join(img_path, name+'_2.png'),im= img_aug2)
    imageio.imwrite(uri =os.path.join(label_path, name+'_1.png'),im= label_aug1)
    imageio.imwrite(uri =os.path.join(label_path, name+'_2.png'),im= label_aug2)
    # ia.imshow(img_aug1)
    # ia.imshow(img_aug2)

    # ia.imshow(label)
    # ia.imshow(label_aug1)
    # ia.imshow(label_aug2)


# name = '1234.png'
# name =  name.split('.')
# print(name[0])
print(len(img_list))