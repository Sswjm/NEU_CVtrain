import os
import sys
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import cv2
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, datasets, transforms
from torchvision.io import read_image
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from PIL import Image
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from sklearn.metrics import f1_score


# 设置相关参数
batch_size = 8
epochs = 100
lr = 5e-4

num_classes = 3
image_size = 512   # 根据efficientnet作者的说明，efficient_b4建议使用的图片尺寸是380

device = "cuda" if torch.cuda.is_available() else "cpu"

labels_map = {
    0: "Image_DCM",
    1: "Image_HCM",
    2: "Image_NOR",
#    4: "Proliferative DR",
}
coef = [0.5, 1.5, 2.5]


def predict_class(value):
  pre = 0
  if value < coef[0]:
    pre = 0
  elif value >= coef[0] and value < coef[1]:
    pre = 1
  elif value >= coef[1] and value < coef[2]:
    pre = 2
  

  return pre

  path_ori = 'data/Heart Data'
path = 'png/Label/'
imagest = []
annotationst = []
imagesv = []
annotationsv = []
for image_path0 in os.listdir(path_ori):
  input_image0 = os.path.join(path_ori, image_path0)
  input_image0 = os.path.join(input_image0, path)
  #print(input_image0)
  patient= []
  train_list = []
  valid_list = []
  if image_path0 == 'Image_DCM' or image_path0 =='Image_HCM' or image_path0 =='Image_NOR':
   input_image01 = os.listdir(input_image0)
   for image_path in  input_image01:
             #print(image_path)
             input_image1 = os.path.join(input_image0, image_path)
             patient.append(input_image1)
   #print(patient)
   train_list, valid_list = train_test_split(patient, test_size=0.2, random_state=42)
   print(train_list)
   print(valid_list)
   #print(train_list)
   for i in train_list:
     for image_path1 in os.listdir(i):
             input_image1 = os.path.join(i, image_path1)
             imagest.append(input_image1)
             if(image_path0 == 'Image_DCM'):
                annotationst.append(0)
             if(image_path0 == 'Image_HCM'):
                annotationst.append(1)
             if(image_path0 == 'Image_NOR'):
                annotationst.append(2)
   for j in valid_list:
     for image_path2 in os.listdir(j):
             input_image2 = os.path.join(j, image_path2)
             imagesv.append(input_image2)
             if(image_path0 == 'Image_DCM'):
                annotationsv.append(0)
             if(image_path0 == 'Image_HCM'):
                annotationsv.append(1)
             if(image_path0 == 'Image_NOR'):
                annotationsv.append(2)


# print(len(imagest))  
# print(len(annotationsv))
# print(len(imagest))  
# print(len(annotationsv))    

# 重载dataset类，以便适应我们的数据集
class HeartDataset(Dataset):
    def __init__(self, image_paths:list, image_labels:list, transform=None):
        self.image_paths = image_paths
        self.img_labels = image_labels
        self.transform = transform
        
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.image_paths[item])#【idx，1】为原始图片，【idx，2】为分割后图片
         
        #image = read_image(img_path)
        #image = Image.open(img_path)
        image = cv2.imread(img_path)
        #image = image.convert("RGB")
        #label = self.img_labels.iloc[idx, 1]
        image = Image.fromarray(np.uint8(image))
        label = self.img_labels[item]
        if self.transform:
            image = self.transform(image)
       
        return image, label

train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        

    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),

    ]
)


train_dataset = HeartDataset(imagest, annotationst, transform=train_transforms)
valid_dataset = HeartDataset(imagesv, annotationsv, transform=test_transforms)



#将五个划分导入五个dataloader中
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

model = models.resnet18(pretrained=False)

in_channel = model.fc.in_features
model.fc = nn.Linear(in_channel, 3)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5,weight_decay=1e-5 )#

#scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.4, patience=3, verbose=True, threshold=0.1, threshold_mode='abs')
#save_path = '/content/drive/MyDrive/best.pth'

epochs_train_loss = []
epochs_val_loss = []
epochs_train_acc = []
epochs_val_acc = []

best_loss = 100.0
best_acc = 0
# Train the model
#total_step = len(train_dataloader)
for epoch in range(epochs):
    # train
    # print('Learning Rate: {}'.format(scheduler.get_last_lr()))
    model.train()
    train_loss = 0.0
    total = 0
    correct = 0
    #一个epoch跑五个不同划分的数据
    for i, (images, labels) in enumerate(tqdm(train_dataloader)):
        images = images.to(device)
        
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('Epoch [{}/{}]' .format(epoch+1, epochs))
    print('Train loss: {:.4f}'.format(train_loss / len(train_dataloader)))
    epochs_train_loss.append(train_loss / len(train_dataloader))
    print('Train Accuracy of the model on the train images: {} %'.format(100 * correct / total))
    epochs_train_acc.append(100 * correct / total)

     # validate the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    val_loss = 0.0
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_dataloader:
            images = images.to(device)
        
          
            labels = labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    
    print('Val loss: {:.4f}'.format(val_loss / len(valid_dataloader)))
    epochs_val_loss.append(val_loss / len(valid_dataloader))
    print('Val Accuracy of the model on the validation images: {} %'.format(100 * correct / total))
    epochs_val_acc.append(100 * correct / total)

    if (val_loss / len(valid_dataloader)) < best_loss:
      best_loss = val_loss / len(valid_dataloader)
    if (100 * correct / total) > best_acc:
      best_acc = 100 * correct / total
      print('Save best model,loss: {:.4f}  acc: {}'.format(best_loss, best_acc))
      #torch.save(model.state_dict(), save_path)

    #torch.save(model.state_dict(), '/content/drive/MyDrive/{}.pth'.format(epoch))
    scheduler.step(val_loss / len(valid_dataloader))