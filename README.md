# NEU-CV-TrainProject

# 本项目为东北大学计算机视觉实训，共三个部分
***数据集为学校项目使用，无法放在此仓库中，请谅解***

**README中仅对相关项目信息进行简单介绍。详细工作见doc文件。

## 一、眼底影像相关
### 1.黄斑中心凹检测
macular文件夹，使用yolov5（https://github.com/ultralytics/yolov5） 训练目标检测模型，并取bounding box的中心点作为黄斑中心凹的位置。
训练过程train.py，详见yolov5官方tutorial。运行检测文件detect.py，检测图片数据位于test/

### 2.眼底疾病分类。使用torchvision中的EfficientNet对眼底疾病图像进行分类。文件为classification.ipynb

### 3.心室分割。使用UNet进行心室分割。

### 4.低剂量CT图像优化。使用CycleGAN进行去噪。
