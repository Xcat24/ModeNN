import torch
import torch.nn.functional as F

def anti_relu(input):
    return - F.relu(-input)