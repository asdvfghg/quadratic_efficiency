import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn.modules import loss
import torch.nn.functional as F

'''
train function for QCNN
'''


def group_parameters(m):
    weight_r, weight_g, weight_b, bias_r, bias_g, bias_b, w, b = [], [], [], [], [], [], [], []
    for name, p in m.named_parameters():
        if 'weight_r' in name:
            weight_r += [p]
        if 'weight_g' in name:
            weight_g += [p]
        if 'weight_b' in name:
            weight_b += [p]
        if 'bias_r' in name:
            bias_r += [p]
        if 'bias_g' in name:
            bias_g += [p]
        if 'bias_b' in name:
            bias_b += [p]
        if 'weight' in name[-6:]:
            w += [p]
        if 'bias' in name[-4:]:
            b += [p]
    return (weight_r, weight_g, weight_b, bias_r, bias_g, bias_b, w, b)

