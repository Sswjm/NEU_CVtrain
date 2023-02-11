import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet18

import cv2
from PIL import Image
import torch.nn as nn

def classification(model, ndarray_img):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_transform = transforms.Compose(
        [transforms.Resize((512, 512)),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    # image = cv2.imread(img_name)
    image = Image.fromarray(np.uint8(ndarray_img))
    image = data_transform(image)
    image = torch.unsqueeze(image,dim=0)
    model.eval()

    with torch.no_grad():

        output = torch.squeeze(model(image.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        pre = torch.argmax(predict).numpy()

    # print('pre : {}'.format(pre))
    

    return pre
    
    



