import torch
import torch.nn.functional as F

def anti_relu(input):
    return - F.relu(-input)

class Poly_3order(torch.nn.Module):
    def forward(self, input):
        out = torch.pow(input,3)
        out = input - (1/3)*out
        return out.clamp(-0.629, 0.629)

class Poly_5order(torch.nn.Module):
    def forward(self, input):
        out_3 = torch.pow(input,3)
        out_5 = torch.pow(input,5)
        out = input - (1/3)*out_3 +(2/15)*out_5
        return out.clamp(-0.673, 0.673)

class Poly_7order(torch.nn.Module):
    def forward(self, input):
        out_3 = torch.pow(input,3)
        out_5 = torch.pow(input,5)
        out_7 = torch.pow(input,7)
        out = input - (1/3)*out_3 + (2/15)*out_5 - (34/630)*out_7
        return out.clamp(-0.661, 0.661)