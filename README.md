# NEU-CV-TrainProject

# 本项目为东北大学计算机视觉实训，共三个部分

## 一、眼底影像相关
### 1.黄斑中心凹检测
macular文件夹，使用yolov5（https://github.com/ultralytics/yolov5） 训练目标检测模型，并取bounding box的中心点作为黄斑中心凹的位置。
训练过程train.py，详见yolov5官方tutorial。运行检测文件detect.py，检测图片数据位于test/
