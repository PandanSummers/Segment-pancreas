import torch
import torch.nn as nn

def dice_coef(output, target):#output为预测结果 target为真实结果
    smooth = 1e-5 #防止0除
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)