import torch
import torchvision
import torchvision.transforms as transforms
from skimage import feature
from skimage.color import rgb2gray

def compute_cnn_out(input_size, kernel_size, padding=(0,0), dilation=(1,1), stride=(1,1), pooling=(2,2)):
    h_out = ((input_size[0]+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)//stride[0] + 1)//pooling[0]
    w_out = ((input_size[1]+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)//stride[1] + 1)//pooling[1]
    return (h_out, w_out)

def compute_5MODE_dim(input_dim):
    temp = 0
    for i in range(1,5):
        temp += len(torch.combinations(torch.rand(input_dim), i+1, with_replacement=True))
    temp += input_dim
    return temp

class pick_edge(object):
    """transform: detect the edge of the image, return 0-1 torch tensor"""
    def __call__(self, pic):
        #handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        temp = torch.zeros(img.shape[:-1])

        if img.shape[-1] == 3:
            temp = torch.tensor(feature.canny(rgb2gray(img), sigma=1.6))
        else:
            temp = torch.tensor(feature.canny(img.reshape(img.shape[:-1]), sigma=1.6))
        temp = torch.where(temp==True, torch.ones(temp.shape), torch.zeros(temp.shape))
        return temp.numpy()

class Pretrain_Select(object):
    '''
    transform: select dims according to the pretrained model's weight
    following the ToTensor transforms
    return 1D tensor
    '''
    def __init__(self, model_path, weight_name='fc.weight', bins_size=9, bins_num=35):
        weight = torch.load(model_path)['state_dict'][weight_name]
        weight = weight.abs().sum(dim=0)/weight.size()[0]
        self.bins_num = bins_num
        self.bins_size = bins_size
        self.topk_index = torch.topk(weight, self.bins_size*self.bins_num)[1]

    def __call__(self, x):
        x = torch.flatten(x, 1)
        return torch.stack([torch.index_select(x, 1, self.topk_index[self.bins_size*i:self.bins_size*i+self.bins_size].to(torch.device('cpu'))) for i in range(self.bins_num)], dim=1)
        


if __name__ == "__main__":
    x = torchvision.datasets.CIFAR10(root='/disk/Dataset/CIFAR-10', train=True, transform=transforms.Compose([pick_edge(), transforms.ToTensor()]), download=True)
    print(x.__getitem__(1)[0].shape)