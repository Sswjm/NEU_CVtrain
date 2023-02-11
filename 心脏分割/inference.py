import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from utils.onehot import onehot_to_mask, mask_to_onehot, array_to_img
from utils.transforms import crop_img, transform
from classification import classification
from models.UNet import UNet

import segmentation_models_pytorch as smp
from torchvision.models import resnet18


def inference(mask, palette, cls_model, ori_w, ori_h, save_path):
    res = onehot_to_mask(np.array(mask.squeeze()).transpose(1,2,0), palette)
    # print(test.shape)
    res = crop_img(res, ori_w, ori_h)
    # (h,w,1)
    cls_res = np.squeeze(res)
    cls_res = np.stack((cls_res,)*3, axis=-1)

    res = array_to_img(res)
    # print(test.size)
    #classification
    # pre = 0
    pre = classification(cls_model, cls_res)
    
    if pre == 0:
        cls = 'DCM'
    elif pre == 1:
        cls = 'HCM'
    else:
        cls = 'NOR'
    
    res.save(save_path + cls + '.png')

def segmentation(test_path='test', save_path='result'):

    try:
        os.mkdir(save_path)
    except:
        print("exists")

    img_size = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    palette = [[0], [85], [170], [255]]
    
    model = smp.UnetPlusPlus(
        encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=4,                      # model output channels (number of classes in your dataset)
    )

    model.to(device)
    model.load_state_dict(torch.load('u++res18_best_0.8693.pth', map_location=device))
    model.eval()
    
    cls_model = resnet18()
    in_channel = cls_model.fc.in_features
    cls_model.fc = nn.Linear(in_channel, 3)
    cls_model.to(device)
    cls_model.load_state_dict(torch.load('best-278-85.0-86.4.pth', map_location=device))

    patient_list = os.listdir(test_path)
    images_list = os.listdir(test_path)

    for patient in tqdm(patient_list):
        images_list = os.listdir(test_path + '/' + patient)
        cur_save_path = save_path + '/' + patient
        try:
            os.mkdir(cur_save_path)
        except:
            print("")
        
        for image_name in images_list:
            image_path = test_path + '/' + patient + '/' + image_name

            image = Image.open(image_path)
            ori_w = image.size[0]
            ori_h = image.size[1]

            image = transform(image, img_size)
            image = np.expand_dims(image, axis=0)
            image = torch.from_numpy(np.array(image, dtype=np.float32))
            image = image.to(device)

            output = model(image)
            output = output.cpu()
            output = output.detach().numpy()
            output = output[0]

            inference(output, palette, cls_model, ori_w, ori_h, cur_save_path + '/' + 'label' + image_name.split('.')[0][5:]+ '-')




