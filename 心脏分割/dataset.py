import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.onehot import mask_to_onehot

# 重载dataset类
# getitem处理流程：读取图片和mask、 由于数据集图像大小不一致，需要进行padding、 转换为(C, H, W) 
class Heart(Dataset):
    def __init__(self, annotations_dataframe, img_dir, img_size, palette, transform=None, target_transform=None):
        self.img_labels = annotations_dataframe
        self.img_dir = img_dir
        self.img_size = img_size
        self.palette = palette
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)

    def preprocess(self, pil_img, pil_mask):

        img_ndarray = np.asarray(pil_img)
        mask_ndarray = np.asarray(pil_mask)
        h, w = img_ndarray.shape[0:2]
        h1, w1 = mask_ndarray.shape[0:2]

        # assert h != h1 or w != w1, 'Image size do not match mask size'

        top = (self.img_size - h) // 2
        left = (self.img_size - w) // 2
        bottom = self.img_size - h - top
        right = self.img_size - w - left

        padding = [top, left, bottom, right]
        
        pixel = 0 if len(img_ndarray.shape) == 2 else (0, 0, 0)

        image_padding = cv2.copyMakeBorder(img_ndarray, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pixel)
        mask_padding = cv2.copyMakeBorder(mask_ndarray, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pixel)
        # if grayscale (image_size, image_size)  if RGB (image_size, image_size, 3)

        # if len(img_ndarray.shape) == 2 and len(mask_ndarray.shape) == 2:  # grayscale
        image = np.expand_dims(image_padding, axis=2)
        mask = np.expand_dims(mask_padding, axis=2)
        # else:
        #   image = image_padding
        #   mask = mask_padding

        mask = mask_to_onehot(mask, self.palette)
        # mask (H, W, num_classes)

        # turn (H, W, C) to （C, H, W）, we can use ToPILImage to do that
        image = image.transpose([2, 0, 1])
        mask = mask.transpose([2, 0, 1])

        image = image / 255
        
        return image, mask, padding

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.loc[idx, 'image'])
        image = Image.open(img_path)

        mask_path = os.path.join(self.img_dir, self.img_labels.loc[idx, 'label'])
        mask = Image.open(mask_path)


        image, mask, padding= self.preprocess(image, mask)  #ndarray

        image = torch.from_numpy(np.array(image, dtype=np.float32))
        mask = torch.from_numpy(np.array(mask, dtype=np.float32))

        return image, mask