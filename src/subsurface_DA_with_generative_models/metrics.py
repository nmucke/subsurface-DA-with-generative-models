import torch
import torch.nn as nn
import torch.nn.functional as F

def RMSE_loss(pred, target):

    loss = 0.0

    for i in range(pred.shape[0]):
        loss += torch.sqrt(torch.mean((pred[i] - target[i])**2))

    return loss / pred.shape[0]

def RRMSE_loss(pred, target):

    loss = 0.0

    for i in range(pred.shape[0]):
        loss += torch.sqrt(torch.mean((pred[i] - target[i])**2)) / torch.sqrt(torch.mean(target[i]**2))

    return loss / pred.shape[0]