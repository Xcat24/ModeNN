from matplotlib.pyplot import axis
import torch
from torch import angle, dtype
import torchvision
import torchvision.transforms as transforms
import layer
from myutils.datasets import ORLdataset, NumpyDataset
from sklearn import datasets
from PIL import ImageDraw, Image
import numpy as np
import math, random, os

# #generate triangle, square, circle patterns, 10*10 pixels each
# bg = np.ones((100,100), dtype=np.uint8)*255
# bg = Image.fromarray(bg)

# draw = ImageDraw.Draw(bg)
# # draw.rectangle([3,3,47,47], 'black', 'black')
# # draw.ellipse([3,3,43,43], 'black', 'black')
# draw.polygon([3,43, 3,3, 43,20], 'black', 'black')

# bg.save('test.jpg')
# bg = np.array(bg)
# print(bg)

#produce circle_square numpy data
# def gen_point():
#     bound = math.sqrt(math.pi/8)
#     x = np.random.uniform(-bound, bound)
#     y = np.random.uniform(-bound, bound)
#     if math.sqrt(x**2 + y**2) > 0.5:
#         label = np.array([0,])
#     else:
#         label = np.array([1,])
    
#     return np.array([[x,y]]), label

# train_data, train_label = gen_point()
# for i in range(999):
#     data, label = gen_point()
#     train_data = np.vstack([train_data, data])
#     train_label = np.concatenate([train_label, label])
# print(np.sum(train_label))

# val_data, val_label = gen_point()
# for i in range(399):
#     data, label = gen_point()
#     val_data = np.vstack([val_data, data])
#     val_label = np.concatenate([val_label, label])
# print(np.sum(val_label))

# np.save('/disk/Dataset/circle_square/pytorch_numpy_data/train_data.npy', train_data)
# np.save('/disk/Dataset/circle_square/pytorch_numpy_data/val_data.npy', val_data)
# np.save('/disk/Dataset/circle_square/pytorch_numpy_data/train_label.npy', train_label)
# np.save('/disk/Dataset/circle_square/pytorch_numpy_data/val_label.npy', val_label)

# #produce playground-like circle_square numpy data
# def gen_point():
#     radius = 5
#     r_pos = np.random.uniform(0, 0.5*radius)
#     angle_pos = np.random.uniform(0, 2*math.pi)
#     pos_x = r_pos*math.sin(angle_pos)
#     pos_y = r_pos*math.cos(angle_pos)
#     r_neg = np.random.uniform(0.7*radius, radius)
#     angle_neg = np.random.uniform(0, 2*math.pi)
#     neg_x = r_neg*math.sin(angle_neg)
#     neg_y = r_neg*math.cos(angle_neg)
#     label = np.array([[1.0,],[-1.0,]], dtype=np.float32)
   
#     return np.array([[pos_x,pos_y],[neg_x,neg_y]], dtype=np.float32), label

# train_data, train_label = gen_point()
# for i in range(499):
#     data, label = gen_point()
#     train_data = np.vstack([train_data, data])
#     train_label = np.vstack([train_label, label])
# print(np.sum(train_label))

# val_data, val_label = gen_point()
# for i in range(199):
#     data, label = gen_point()
#     val_data = np.vstack([val_data, data])
#     val_label = np.vstack([val_label, label])
# print(np.sum(val_label))
# print(train_label.shape)
# print(val_label.shape)
# np.save('/disk/Dataset/circle_square/playground_like/train_data.npy', train_data)
# np.save('/disk/Dataset/circle_square/playground_like/val_data.npy', val_data)
# np.save('/disk/Dataset/circle_square/playground_like/train_label.npy', train_label.reshape((-1,1)))
# np.save('/disk/Dataset/circle_square/playground_like/val_label.npy', val_label.reshape((-1,1)))

# #produce T-C numpy data
# def gen_pixel():
#     return 0.25*np.random.random() + 0.25

# def produce_t():
#     t_1 = np.array([[1., 1., 1.],[-1., 1., -1.],[-1., 1., -1.]])
#     t_2 = np.array([[-1., 1., -1.],[-1., 1., -1.],[1., 1., 1.]])
#     for t in [t_1, t_2, t_1.T, t_2.T]:
#         for i in range(9):
#             t[i//3][i%3] = gen_pixel()*t[i//3][i%3]
#     return np.array([t_1, t_2, t_1.T, t_2.T])

# def produce_c():
#     c_1 = np.array([[-1., 1., 1.],[-1., 1., -1.],[-1., 1., 1.]])
#     c_2 = np.array([[1., 1., -1.],[-1., 1., -1.],[1., 1., -1.]])
#     for c in [c_1, c_2, c_1.T, c_2.T]:
#         for i in range(9):
#             c[i//3][i%3] = gen_pixel()*c[i//3][i%3]
#     return np.array([c_1, c_2, c_1.T, c_2.T])

# train_data = produce_t()
# for _ in range(124):
#     train_data = np.vstack([train_data, produce_t()])

# for _ in range(125):
#     train_data = np.vstack([train_data, produce_c()])

# train_label = np.concatenate((np.zeros(500, dtype=np.int), np.ones(500, dtype=np.int)))


# val_data = produce_t()
# for _ in range(49):
#     val_data = np.vstack([val_data, produce_t()])

# for _ in range(50):
#     val_data = np.vstack([val_data, produce_c()])

# val_label = np.concatenate((np.zeros(200, dtype=np.int), np.ones(200, dtype=np.int)))

# np.save('/disk/Dataset/T-C/pytorch_numpy_data/train_data.npy', train_data)
# np.save('/disk/Dataset/T-C/pytorch_numpy_data/val_data.npy', val_data)
# np.save('/disk/Dataset/T-C/pytorch_numpy_data/train_label.npy', train_label)
# np.save('/disk/Dataset/T-C/pytorch_numpy_data/val_label.npy', val_label)

# #produce XOR numpy data
# num = 500
# train_pos = np.vstack((np.stack((np.random.random(250), np.random.random(250))).T, np.stack((-np.random.random(250), -np.random.random(250))).T))
# train_neg = np.vstack((np.stack((-np.random.random(250), np.random.random(250))).T, np.stack((np.random.random(250), -np.random.random(250))).T))
# train_data = np.vstack((train_pos, train_neg))
# train_label = np.concatenate((np.zeros(500, dtype=np.int), np.ones(500, dtype=np.int)))

# val_pos = np.vstack((np.stack((np.random.random(100), np.random.random(100))).T, np.stack((-np.random.random(100), -np.random.random(100))).T))
# val_neg = np.vstack((np.stack((-np.random.random(100), np.random.random(100))).T, np.stack((np.random.random(100), -np.random.random(100))).T))
# val_data = np.vstack((val_pos, val_neg))
# val_label = np.concatenate((np.zeros(200, dtype=np.int), np.ones(200, dtype=np.int)))
# np.save('/disk/Dataset/XOR/pytorch_numpy_data/train_data.npy', train_data)
# np.save('/disk/Dataset/XOR/pytorch_numpy_data/val_data.npy', val_data)
# np.save('/disk/Dataset/XOR/pytorch_numpy_data/train_label.npy', train_label)
# np.save('/disk/Dataset/XOR/pytorch_numpy_data/val_label.npy', val_label)

# #4points xor data
# train_data = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
# train_label = np.array([0, 0, 1, 1], dtype=np.int)
# val_data = train_data
# val_label = train_label
# np.save('/disk/Dataset/XOR/4points_crossentropy/train_data.npy', train_data)
# np.save('/disk/Dataset/XOR/4points_crossentropy/val_data.npy', val_data)
# np.save('/disk/Dataset/XOR/4points_crossentropy/train_label.npy', train_label)
# np.save('/disk/Dataset/XOR/4points_crossentropy/val_label.npy', val_label)

# #4points xor data
# train_data = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
# train_label = np.array([[1.0,], [1.0], [-1.0,], [-1.0]], dtype=np.float32)
# val_data = train_data
# val_label = train_label
# np.save('/disk/Dataset/XOR/4points_mse/train_data.npy', train_data)
# np.save('/disk/Dataset/XOR/4points_mse/val_data.npy', val_data)
# np.save('/disk/Dataset/XOR/4points_mse/train_label.npy', train_label)
# np.save('/disk/Dataset/XOR/4points_mse/val_label.npy', val_label)

# #tf-playground-like XOR data for CrossEntropy
# num = 500
# train_pos = np.vstack((np.stack((np.random.randint(100,500,(250,))/100, np.random.randint(100,500,(250,))/100)).T, np.stack((-np.random.randint(100,500,(250,))/100, -np.random.randint(100,500,(250,))/100)).T))
# train_neg = np.vstack((np.stack((-np.random.randint(100,500,(250,))/100, np.random.randint(100,500,(250,))/100)).T, np.stack((np.random.randint(100,500,(250,))/100, -np.random.randint(100,500,(250,))/100)).T))
# train_data = np.vstack((train_pos, train_neg))
# train_label = np.concatenate((np.zeros(500, dtype=np.int), np.ones(500, dtype=np.int)))

# val_pos = np.vstack((np.stack((np.random.randint(100,500,(100,))/100, np.random.randint(100,500,(100,))/100)).T, np.stack((-np.random.randint(100,500,(100,))/100, -np.random.randint(100,500,(100,))/100)).T))
# val_neg = np.vstack((np.stack((-np.random.randint(100,500,(100,))/100, np.random.randint(100,500,(100,))/100)).T, np.stack((np.random.randint(100,500,(100,))/100, -np.random.randint(100,500,(100,))/100)).T))
# val_data = np.vstack((val_pos, val_neg))
# val_label = np.concatenate((np.zeros(200, dtype=np.int), np.ones(200, dtype=np.int)))
# print(val_data.shape)
# print(val_label.shape)
# np.save('/disk/Dataset/XOR/tf_playground_like_crossentropy/train_data.npy', train_data)
# np.save('/disk/Dataset/XOR/tf_playground_like_crossentropy/val_data.npy', val_data)
# np.save('/disk/Dataset/XOR/tf_playground_like_crossentropy/train_label.npy', train_label)
# np.save('/disk/Dataset/XOR/tf_playground_like_crossentropy/val_label.npy', val_label)

# #tf-playground-like XOR data
# num = 500
# train_pos = np.vstack((np.stack((np.random.randint(100,500,(250,))/100, np.random.randint(100,500,(250,))/100)).T, np.stack((-np.random.randint(100,500,(250,))/100, -np.random.randint(100,500,(250,))/100)).T))
# train_neg = np.vstack((np.stack((-np.random.randint(100,500,(250,))/100, np.random.randint(100,500,(250,))/100)).T, np.stack((np.random.randint(100,500,(250,))/100, -np.random.randint(100,500,(250,))/100)).T))
# train_data = np.vstack((train_pos, train_neg))
# train_label = np.concatenate((np.ones(500, dtype=np.float32), np.zeros(500, dtype=np.float32) - 1)).reshape((-1,1))

# val_pos = np.vstack((np.stack((np.random.randint(100,500,(100,))/100, np.random.randint(100,500,(100,))/100)).T, np.stack((-np.random.randint(100,500,(100,))/100, -np.random.randint(100,500,(100,))/100)).T))
# val_neg = np.vstack((np.stack((-np.random.randint(100,500,(100,))/100, np.random.randint(100,500,(100,))/100)).T, np.stack((np.random.randint(100,500,(100,))/100, -np.random.randint(100,500,(100,))/100)).T))
# val_data = np.vstack((val_pos, val_neg))
# val_label = np.concatenate((np.ones(200, dtype=np.float32), np.zeros(200, dtype=np.float32) - 1)).reshape((-1,1))
# print(val_data.shape)
# print(val_label.shape)
# np.save('/disk/Dataset/XOR/tf_playground_like/train_data.npy', train_data)
# np.save('/disk/Dataset/XOR/tf_playground_like/val_data.npy', val_data)
# np.save('/disk/Dataset/XOR/tf_playground_like/train_label.npy', train_label)
# np.save('/disk/Dataset/XOR/tf_playground_like/val_label.npy', val_label)

# #optimized XOR data
# num = 500
# train_pos = np.vstack((np.stack((np.random.randint(100,900,(250,))/1000, np.random.randint(100,900,(250,))/1000)).T, np.stack((-np.random.randint(100,900,(250,))/1000, -np.random.randint(100,900,(250,))/1000)).T))
# train_neg = np.vstack((np.stack((-np.random.randint(100,900,(250,))/1000, np.random.randint(100,900,(250,))/1000)).T, np.stack((np.random.randint(100,900,(250,))/1000, -np.random.randint(100,900,(250,))/1000)).T))
# train_data = np.vstack((train_pos, train_neg))
# train_label = np.concatenate((np.zeros(500, dtype=np.int), np.ones(500, dtype=np.int))).reshape((-1,1))

# val_pos = np.vstack((np.stack((np.random.randint(100,900,(100,))/1000, np.random.randint(100,900,(100,))/1000)).T, np.stack((-np.random.randint(100,900,(100,))/1000, -np.random.randint(100,900,(100,))/1000)).T))
# val_neg = np.vstack((np.stack((-np.random.randint(100,900,(100,))/1000, np.random.randint(100,900,(100,))/1000)).T, np.stack((np.random.randint(100,900,(100,))/1000, -np.random.randint(100,900,(100,))/1000)).T))
# val_data = np.vstack((val_pos, val_neg))
# val_label = np.concatenate((np.zeros(200, dtype=np.int), np.ones(200, dtype=np.int))).reshape((-1,1))
# print(val_data.shape)
# print(val_label.shape)
# np.save('/disk/Dataset/XOR/optimized/train_data.npy', train_data)
# np.save('/disk/Dataset/XOR/optimized/val_data.npy', val_data)
# np.save('/disk/Dataset/XOR/optimized/train_label.npy', train_label)
# np.save('/disk/Dataset/XOR/optimized/val_label.npy', val_label)

#produce Iris numpy data for mse loss
# iris=datasets.load_iris()
# irisFeature=iris.data
# irisTarget=iris.target
# print(iris.data.shape)
# print(iris.data[:,0].max())
# for i in range(4):
#     iris.data[:,i] = iris.data[:,i]/iris.data[:,i].max()
# print(iris.data.max())
# print(iris.target)
# val_data = np.vstack((iris.data[35:50], iris.data[85:100], iris.data[135:150]))
# train_data = np.vstack((iris.data[0:35], iris.data[50:85], iris.data[100:135]))
# train_label = np.vstack((np.repeat(np.array([[1., 0., 0.]], dtype=np.float32), 35, axis=0), np.repeat(np.array([[0., 1., 0.]], dtype=np.float32), 35, axis=0), np.repeat(np.array([[0., 0., 1.]], dtype=np.float32), 35, axis=0)))
# val_label = np.vstack((np.repeat(np.array([[1., 0., 0.]], dtype=np.float32), 15, axis=0), np.repeat(np.array([[0., 1., 0.]], dtype=np.float32), 15, axis=0), np.repeat(np.array([[0., 0., 1.]], dtype=np.float32), 15, axis=0)))

# np.save('/disk/Dataset/Iris/scaled_mse/train_data.npy', train_data)
# np.save('/disk/Dataset/Iris/scaled_mse/val_data.npy', val_data)
# np.save('/disk/Dataset/Iris/scaled_mse/train_label.npy', train_label)
# np.save('/disk/Dataset/Iris/scaled_mse/val_label.npy', val_label)

#produce Iris numpy data
iris=datasets.load_iris()
irisFeature=iris.data
irisTarget=iris.target
print(iris.data.shape)
print(iris.data[:,0].max())
for i in range(4):
    iris.data[:,i] = iris.data[:,i]/iris.data[:,i].max()
print(iris.data.max())
for i in range(6):
    train_num = 30-5*i
    # random_idx = random.sample(range(45), train_num)
    random_idx = [2, 17, 39, 22, 3, 13, 10, 42, 33, 20, 12, 19, 23, 41, 8, 35, 29, 40, 30, 21, 43, 44, 32, 0, 36, 31, 11, 28, 34, 14, 7, 27, 38, 1, 5]
    random_idx = random_idx[:train_num]
    random_idx = random_idx + [_+50 for _ in random_idx] + [_+100 for _ in random_idx]

    train_data = np.vstack([iris.data[_,:] for _ in random_idx])
    train_label = np.array([iris.target[_] for _ in random_idx])
    val_data = np.delete(iris.data, random_idx, axis=0)
    val_label = np.delete(iris.target, random_idx)

    # val_data = np.vstack((iris.data[train_num:50], iris.data[50+train_num:100], iris.data[100+train_num:150]))
    # train_data = np.vstack((iris.data[0:train_num], iris.data[50:50+train_num], iris.data[100:100+train_num]))
    # train_label = np.concatenate((iris.target[0:train_num], iris.target[50:50+train_num], iris.target[100:100+train_num]))
    # val_label = np.concatenate((iris.target[train_num:50], iris.target[50+train_num:100], iris.target[100+train_num:150]))
    print(train_data.shape)
    print(val_data.shape)
    print(train_label.shape)
    print(val_label.shape)
    path = '/disk/Dataset/Iris/scaled/random1/'+str(train_num)+'train/'+str(3*(35-train_num)+45)+'val/'
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path+'train_data.npy', train_data)
    np.save(path+'val_data.npy', val_data)
    np.save(path+'train_label.npy', train_label)
    np.save(path+'val_label.npy', val_label)

np.save('/disk/Dataset/Iris/scaled/train_data.npy', train_data)
np.save('/disk/Dataset/Iris/scaled/val_data.npy', val_data)
np.save('/disk/Dataset/Iris/scaled/train_label.npy', train_label)
np.save('/disk/Dataset/Iris/scaled/val_label.npy', val_label)

np.save('/disk/Dataset/Iris/pytorch_numpy_data/train_data.npy', train_data)
np.save('/disk/Dataset/Iris/pytorch_numpy_data/val_data.npy', val_data)
np.save('/disk/Dataset/Iris/pytorch_numpy_data/train_label.npy', train_label)
np.save('/disk/Dataset/Iris/pytorch_numpy_data/val_label.npy', val_label)


mnist_train_data = torchvision.datasets.MNIST(root='/disk/Dataset/', train=True, transform=transforms.Compose([transforms.ToTensor()]))
mnist_val_data = torchvision.datasets.MNIST(root='/disk/Dataset/', train=False, transform=transforms.Compose([transforms.ToTensor()]))

temp = torch.utils.data.DataLoader(dataset=mnist_train_data, batch_size=60000, shuffle=False)
for i , j in enumerate(temp):
    index, train_data = i, j

temp = torch.utils.data.DataLoader(dataset=mnist_val_data, batch_size=10000, shuffle=False)
for i , j in enumerate(temp):
    index, val_data = i, j

slconv = layer.SLConv(in_channel=1, stride=1, padding=1)
pool = torch.nn.MaxPool2d(4)

# x = mnist_train_data.__getitem__(0)[0].unsqueeze(1)
out = slconv(val_data[0])


root = '/disk/Dataset'
# cifar_train = torchvision.datasets.CIFAR10('/disk/Dataset/CIFAR-10', train=True, download=True)
# cifar_test = torchvision.datasets.CIFAR10('/disk/Dataset/CIFAR-10', train=False, download=True)

imagenet_train = torchvision.datasets.ImageNet('/disk/Dataset/ImageNet', split='train', download=True)
imagenet_val = torchvision.datasets.ImageNet('/disk/Dataset/ImageNet', split='val', download=True)