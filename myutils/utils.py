import torch

def compute_cnn_out(input_size, kernel_size, padding=(0,0), dilation=(1,1), stride=(1,1), pooling=(2,2)):
    h_out = ((input_size[0]+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)//stride[0] + 1)//pooling[0]
    w_out = ((input_size[1]+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)//stride[1] + 1)//pooling[1]
    return (h_out, w_out)