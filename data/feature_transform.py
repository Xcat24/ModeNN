import torch
import cv2
import math
import numpy as np

class SVD_transform(object):
    def __call__(self, sample):
        u,s,v = torch.svd(sample)
        return s

class DE_transform(object):
    def __init__(self, order=2):
        self.order = order

    def __call__(self, sample):
        sample = sample.view((sample.shape[0], -1))
        de = torch.cat([torch.stack([torch.prod(torch.combinations(a, i+1, with_replacement=True), dim=1) for a in sample]) for i in range(self.order)], dim=-1)
        return de.squeeze()

class RandomSampleDE_transform(object):
    def __init__(self, input_size_x, input_size_y, group_num, order):
        self.order = order
        self.group_num = group_num
        self.x = torch.normal(mean=input_size_x/2, std=4, size=(group_num, 3)).long().clamp(0,input_size_x-1)
        self.y = torch.normal(mean=input_size_y/2, std=4, size=(group_num, 3)).long().clamp(0,input_size_y-1)

    def __call__(self, sample):
        assert len(sample.shape) == 3
        assert isinstance(sample, torch.Tensor)
        origin = torch.flatten(sample, 1)
        result = []
        for i in range(self.group_num):
            tmp = torch.index_select(sample, dim=1, index=self.x[i])
            tmp = torch.index_select(tmp, dim=2, index=self.y[i])
            result.append(tmp)
        random_sample = torch.stack(result, dim=1).view(-1, 9)
        de = torch.cat([torch.stack([torch.prod(torch.combinations(a, i+2, with_replacement=True), dim=1) for a in random_sample]) for i in range(self.order - 1)], dim=-1)
        de = de.squeeze().view(sample.shape[0],-1)
        return torch.cat([origin, de], dim=-1).squeeze()

class DCT_transform(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, sample):
        assert len(sample.shape) == 3
        tmp = sample.numpy()
        tmp = [torch.tensor(cv2.dct(x))[:self.dim, :self.dim] for x in tmp]
        return torch.stack(tmp, dim=0)


class RandomPick_transform(object):
    def __init__(self, input_dim=784, output_dim=64):
        self.mask = torch.randint(input_dim, (output_dim,))

    def __call__(self, sample):
        origin = torch.flatten(sample, 1)
        selected = torch.index_select(origin,dim=1,index=self.mask)
        return (origin, selected)

class HOG_transform(object):
    def __init__(self, cell_x, cell_y, cell_w):
        self.cell_x = cell_x
        self.cell_y = cell_y
        self.cell_w = cell_w
    
    def render_gradient(self, image, cell_gradient):
        cell_size  = 16
        bin_size = 9
        angle_unit = 180// bin_size
        cell_width =  cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(abs(magnitude))))
                    angle += angle_gap
        return image
    
    def div(self, img, cell_x, cell_y, cell_w):
        cell = np.zeros(shape=(cell_x, cell_y, cell_w, cell_w))
        img_x = np.split(img, cell_x , axis=0)
        for i in range(cell_x):
            img_y = np.split(img_x[i], cell_y, axis=1)
            for j in range(cell_y):
                cell[i][j] = img_y[j]
        return cell

    def get_bins(self, grad_cell, ang_cell):
        bins = np.zeros(shape=(grad_cell.shape[0], grad_cell.shape[1], 9))
        for i in range(grad_cell.shape[0]):
            for j in range(grad_cell.shape[1]):
                binn = np.zeros(9)
                grad_list = np.int8(grad_cell[i, j].flatten())  
                ang_list = ang_cell[i, j].flatten()  
                ang_list = np.int8(ang_list / 20.0)  
                ang_list[ang_list >= 9] = 0
                for m in range(len(ang_list)):
                    binn[ang_list[m]] += int(grad_list[m])  
                bins[i][j] = binn

        return bins

    def hog(self, img, cell_x, cell_y, cell_w):
        gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  
        gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  
        gradient_magnitude = np.sqrt(np.power(gradient_values_x, 2) + np.power(gradient_values_y, 2))
        gradient_angle = np.arctan2(gradient_values_x, gradient_values_y)
        gradient_angle[gradient_angle > 0] *= 180 / np.pi
        gradient_angle[gradient_angle < 0] = (gradient_angle[gradient_angle < 0] + np.pi) * 180 / np.pi

        grad_cell = self.div(gradient_magnitude, cell_x, cell_y, cell_w)
        ang_cell = self.div(gradient_angle, cell_x, cell_y, cell_w)
        bins = self.get_bins(grad_cell, ang_cell)
        hog_image = self.render_gradient(np.zeros([img.shape[0], img.shape[1]]), bins) #会归一化 bins 

        feature = []
        for i in range(cell_x - 1):
            for j in range(cell_y - 1):
                tmp = []
                tmp.append(bins[i, j])
                tmp.append(bins[i + 1, j])
                tmp.append(bins[i, j + 1])
                tmp.append(bins[i + 1, j + 1])
                tmp -= np.mean(tmp)
                feature.append(tmp.flatten())
        return np.array(feature).flatten()

    def __call__(self, sample):
        if len(sample.shape) == 2:
            return torch.tensor(self.hog(sample.numpy(), self.cell_x, self.cell_y, self.cell_w), dtype=torch.float32)
        elif len(sample.shape) == 3:
            if sample.shape[0] == 1:
                return torch.tensor(self.hog(torch.squeeze(sample).numpy(), self.cell_x, self.cell_y, self.cell_w), dtype=torch.float32)
            elif sample.shape[0] == 3:
                #TODO transform to gray format
                pass
            else:
                #TODO report error
                pass
        else:
            #TODO report error
            pass
