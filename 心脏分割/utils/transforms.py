import numpy as np
import cv2
import torch 

def crop_img(img, x, y):
    h = (256 - x) // 2
    w = (256 - y) // 2

    hh = h + x
    ww = w + y
    hh = int(hh)
    ww = int(ww)
    h = int(h)
    w = int(w)
    img = img[w : ww, h : hh]
    return img

def transform(pil_img, img_size):
    img_ndarray = np.asarray(pil_img)
    h, w = img_ndarray.shape[0:2]

    top = (img_size - h) // 2
    left = (img_size - w) // 2
    bottom = img_size - h - top
    right = img_size - w - left
    
    image_padding = cv2.copyMakeBorder(img_ndarray, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    image = np.expand_dims(image_padding, axis=2)
    image = image.transpose([2, 0, 1])

    image = image / 255

    return image