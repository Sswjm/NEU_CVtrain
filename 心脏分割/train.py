import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, datasets, transforms

import segmentation_models_pytorch as smp

from sklearn.model_selection import train_test_split

from tqdm import tqdm
from dataset import Heart
from utils.metrics import *
from utils.loss import *



# 设置训练参数
batch_size = 16
epochs = 100
lr = 5e-4

image_size = 256   

set_palette = [[0], [85], [170], [255]]  # 四分类，背景：0，其余三个部位的像素值分贝为85,170,255
# one-hot 0: [1 0 0 0] 85: [0 1 0 0] 170: [0 0 1 0] 255: [0 0 0 1] 
num_classes = 4

device = "cuda" if torch.cuda.is_available() else "cpu"

# 设置path参数，导入csv标签，划分训练集和验证集
input_path = '/content/drive/MyDrive/HeartData'
# input_path = 'HeartData'
# 读取csv annotation file
df_train = pd.read_csv(input_path + '/train.csv')

train_list, valid_list = train_test_split(df_train, test_size=0.2, random_state=2022)
train_list.index = list(range(len(train_list)))
valid_list.index = list(range(len(valid_list)))

# data enhancement
train_transform = transforms.Compose(
    [
    # transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.repeat(3,1,1)),
    ]
)

mask_transform = transforms.Compose(
    [
    # transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.repeat(3,1,1)),
    ]
)

train_dataset = Heart(train_list, input_path, img_size=image_size, palette=set_palette)
valid_dataset = Heart(valid_list, input_path, img_size=image_size, palette=set_palette)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

model = smp.UnetPlusPlus(
    encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=4,                      # model output channels (number of classes in your dataset)
)

model.to(device)

# 使用dice系数进行训练

# criterion = SoftDiceLoss(num_classes)
criterion = BCE_Dice_Loss(num_classes)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=3, verbose=True, threshold=0.1, threshold_mode='abs')

for epoch in range(epochs):
  train_class_dices = np.array([0] * (num_classes - 1), dtype=np.float32)
  val_class_dices = np.array([0] * (num_classes - 1), dtype=np.float32)
  train_dice_arr = []
  val_dice_arr = []
  train_losses = []
  val_losses = []
  best_dice = 0.0

  model.train()
  for i, (images, masks) in enumerate(tqdm(train_dataloader)):
    images = images.to(device)
    masks = masks.to(device)

    # forward
    outputs = model(images)
    # outputs = torch.sigmoid(outputs)

    loss = criterion(outputs, masks)
    train_losses.append(loss.item())

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    class_dice = []
    for i in range(1, num_classes):
      cur_dice = diceCoeffv2(outputs[:, i:i + 1, :], masks[:, i:i + 1, :]).cpu().item()
      class_dice.append(cur_dice)

    mean_dice = sum(class_dice) / len(class_dice)
    train_dice_arr.append(class_dice)
    train_class_dices += np.array(class_dice)

  train_loss = np.average(train_losses)
  train_dice_arr = np.array(train_dice_arr)
  train_class_dices = train_class_dices / len(train_dataloader)
  train_mean_dice = train_class_dices.sum() / train_class_dices.size
  iou = train_mean_dice / (2 - train_mean_dice)
  print('epoch [{}/{}], train_loss: {:.4}, train_mean_dice: {:.4}, iou: {:.4}, dice_class1: {:.4}, dice_class2: {:.4}, dice_class3: {:.4}'
    .format(epoch+1, epochs, train_loss, train_mean_dice, iou, train_class_dices[0], train_class_dices[1], train_class_dices[2]))
  
  model.eval()
  with torch.no_grad():
    for images, masks in valid_dataloader:
      images = images.to(device)
      masks = masks.to(device)
      outputs = model(images)
      # outputs = torch.sigmoid(outputs)
      
      loss = criterion(outputs, masks)

      val_losses.append(loss.item())
      # outputs = outputs.cpu().detach()
      val_class_dice = []
      for i in range(1, num_classes):
        cur_dice = diceCoeffv2(outputs[:, i:i + 1, :], masks[:, i:i + 1, :]).cpu().item()
        val_class_dice.append(cur_dice)

      val_dice_arr.append(val_class_dice)
      val_class_dices += np.array(val_class_dice)

    val_loss = np.average(val_losses)
    val_dice_arr = np.array(val_dice_arr)
    val_class_dices = val_class_dices / len(valid_dataloader)
    val_mean_dice = val_class_dices.sum() / val_class_dices.size
    iou = val_mean_dice/ (2 - val_mean_dice)
    print('epoch [{}/{}], val_loss: {:.4}, val_mean_dice: {:.4}, iou: {:.4}, dice_class1: {:.4}, dice_class2: {:.4}, dice_class3: {:.4}'
      .format(epoch+1, epochs, val_loss, val_mean_dice, iou, val_class_dices[0], val_class_dices[1], val_class_dices[2]))
    if(val_mean_dice > best_dice):
      best_dice = val_mean_dice
      torch.save(model.state_dict(), 'best.pth')

    
  scheduler.step(val_loss)