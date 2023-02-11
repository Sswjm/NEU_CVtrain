# metrics: Dice Loss and IoU
import torch
from torch import nn
import numpy as np


def diceCoeffv2(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """
    
    if activation is None or activation == 'none':
      activation_fn = lambda x:x
    elif activation == 'sigmoid':
      activation_fn = nn.Sigmoid()
    elif activation == 'softmax2d':
      activation_fn = nn.Softmax2d()
    else:
      raise NotImplementedError("Activation implemented for sigmoid or softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    score = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return score.sum() / N

def precision(pred, gt, eps=1e-5):
    """TP / (TP + FP)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))

    score = (tp.float() + eps) / ((tp + fp).float() + eps)

    return score.sum() / N


def sensitivity(pred, gt, eps=1e-5):
    """TP / (TP + FN)"""
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0))

    score = (tp.float() + eps) / ((tp + fn).float() + eps)

    return score.sum() / N

def recall(pred, gt, eps=1e-5):
    return sensitivity(pred, gt)