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


# for i in tqdm(img_list):
#     a=list(i)
#     if '_' in a:
#         print(i)
#         os.remove(os.path.join(img_path, i))

# for j in tqdm(label_list):
#     a= list(j)
#     if '_' in a:
#         print(j)
#         os.remove(os.path.join(label_path,j))
print(len(img_list), len(label_list))