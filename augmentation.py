import torch
import numpy as np
import random
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from PIL import Image
import cv2
import kornia


def baseline(imgs, labels, args):
    two_hot_labels = torch.zeros((imgs.shape[0], args.num_class))
    two_hot_labels[np.arange(imgs.shape[0]), labels[0:imgs.shape[0]]] = 1.0

    if imgs.is_cuda:
        two_hot_labels = two_hot_labels.cuda(args.gpu)

    return imgs, two_hot_labels

def MixUp(imgs, labels, args):
    imgs = imgs.cuda(args.gpu, non_blocking=True)
    two_hot_labels = torch.zeros((imgs.shape[0], args.num_class)).cuda(args.gpu, non_blocking=True)
    
    if torch.rand(1) <= args.prob:

        rand_index = torch.randperm(imgs.size()[0]).cuda()

        new_imgs = imgs[rand_index,:,:,:]
        new_labels = labels[rand_index]

        lamda = torch.rand(1).cuda(args.gpu, non_blocking=True)
        new_imgs = lamda*imgs + (1-lamda)*new_imgs
        
        two_hot_labels[np.arange(imgs.shape[0]), labels] = lamda
        two_hot_labels[np.arange(imgs.shape[0]), new_labels] += (1 - lamda)

        return new_imgs, two_hot_labels
    else:
        two_hot_labels[np.arange(imgs.shape[0]), labels[0:imgs.shape[0]]] = 1.0

        return imgs, two_hot_labels
def rand_bbox(lam, W, H):
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def CutMix(imgs, labels, args):
    imgs = imgs.cuda(args.gpu, non_blocking=True)
    two_hot_labels = torch.zeros((imgs.shape[0], args.num_class)).cuda(args.gpu, non_blocking=True)
    
    if torch.rand(1) <= args.prob:

        image_size = imgs.shape[3]
        rand_index = torch.randperm(imgs.size()[0]).cuda()

        new_imgs = imgs[rand_index,:,:,:]
        new_labels = labels[rand_index]

        alpha = args.alpha
        beta = args.beta
        bbx1, bby1, bbx2, bby2 = rand_bbox(np.random.beta(alpha, beta), image_size, image_size)
        imgs[:,:,bbx1:bbx2, bby1:bby2] = new_imgs[:,:,bbx1:bbx2, bby1:bby2]
        ratio = (bbx2 - bbx1)/image_size*(bby2 - bby1)/image_size

        two_hot_labels[np.arange(imgs.shape[0]), labels] += 1 - ratio
        two_hot_labels[np.arange(imgs.shape[0]), new_labels] += ratio
        
        return imgs, two_hot_labels
    else:
        two_hot_labels[np.arange(imgs.shape[0]), labels[0:imgs.shape[0]]] = 1.0
        return imgs, two_hot_labels


def saliency_bbox(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
    
def SaliencyMix(imgs, labels, args):
    two_hot_labels = torch.zeros((imgs.shape[0], args.num_class)).cuda(args.gpu, non_blocking=True)

    if torch.rand(1) <= args.prob:
        input = imgs
        lam = np.random.beta(args.alpha, args.beta)
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = labels
        target_b = labels[rand_index]
        bbx1, bby1, bbx2, bby2 = saliency_bbox(input[rand_index[0]], lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        # compute output
        two_hot_labels[np.arange(imgs.shape[0]), target_a] = lam
        two_hot_labels[np.arange(imgs.shape[0]), target_b] += 1 - lam
        return input, two_hot_labels
    else:
        two_hot_labels[np.arange(imgs.shape[0]), labels[0:imgs.shape[0]]] = 1.0
        return imgs, two_hot_labels
    
def gkern_torch(kernel_size, u_w, u_h, sigma, args):
    kernel_size = kernel_size
    s = kernel_size * 2
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(s)
    x_grid = x_cord.repeat(s).view(s, s)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).cuda(args.gpu, non_blocking=True)

    xy_grid = torch.roll(xy_grid, u_w, 0)
    xy_grid = torch.roll(xy_grid, u_h, 1)
    crop_size = s // 4
    xy_grid = xy_grid[crop_size: s - crop_size, crop_size: s - crop_size]

    mean = (s - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    
    #(1./(2.*math.pi*variance)) *\
    gaussian_kernel = torch.exp(
                        -torch.sum((xy_grid - mean)**2., dim=-1) /\
                        (2*variance)
                    )
    # Make sure sum of values in gaussian kernel equals 1.
    #gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(kernel_size, kernel_size)
    
    return gaussian_kernel

def SmoothMix(imgs, labels, args):
    imgs = imgs.cuda(args.gpu, non_blocking=True)
    two_hot_labels = torch.zeros((imgs.shape[0], args.num_class)).cuda(args.gpu, non_blocking=True)
    
    if torch.rand(1) <= args.prob:
        image_size = imgs.shape[3]
        rand_index = torch.randperm(imgs.size()[0]).cuda()

        new_imgs = imgs[rand_index,:,:,:]
        new_labels = labels[rand_index]

        u_w = torch.randint(0, image_size, (1,)).cuda(args.gpu, non_blocking=True) - image_size/2
        u_h = torch.randint(0, image_size, (1,)).cuda(args.gpu, non_blocking=True) - image_size/2

        sigma = ((torch.rand(1)/4 + 0.25) * image_size).cuda(args.gpu, non_blocking=True)
        kernel = gkern_torch(image_size, int(u_w), int(u_h), sigma, args).cuda(args.gpu, non_blocking=True)
        
        ratio = torch.sum(kernel)/image_size/image_size
        
        kernel = kernel.repeat(imgs.shape[0], 3, 1, 1)
        imgs = imgs*(1-kernel) + new_imgs*(kernel)    
        
        two_hot_labels[np.arange(imgs.shape[0]), labels] = 1 - ratio
        two_hot_labels[np.arange(imgs.shape[0]), new_labels] += ratio

        return imgs, two_hot_labels
    else:
        two_hot_labels[np.arange(imgs.shape[0]), labels[0:imgs.shape[0]]] = 1.0
        return imgs, two_hot_labels

def ResizeMix(imgs, labels, args):
    imgs = imgs.cuda(args.gpu, non_blocking=True)
    two_hot_labels = torch.zeros((imgs.shape[0], args.num_class)).cuda(args.gpu, non_blocking=True)
    
    if torch.rand(1) <= args.prob:
        image_size = imgs.shape[3]
        rand_index = torch.randperm(imgs.size()[0]).cuda()

        new_imgs = imgs[rand_index,:,:,:]
        new_labels = labels[rand_index]

        a = args.resize_lower
        b = args.resize_upper
        scale_rate = (b - a) * torch.rand(1) + a
        
        new_size = int(torch.ceil(scale_rate * image_size))

        new_imgs = torchvision.transforms.functional.resize(new_imgs, new_size)

        start_point_x = torch.randint(1, image_size-new_size-1, (1,)).cuda(args.gpu, non_blocking=True)
        start_point_y = torch.randint(1, image_size-new_size-1, (1,)).cuda(args.gpu, non_blocking=True)

        imgs[:,:,start_point_x:start_point_x + new_size, start_point_y:start_point_y + new_size] = new_imgs[:,:,:,:]
        ratio = (new_size/image_size)*(new_size/image_size)
        
        two_hot_labels[np.arange(imgs.shape[0]), labels] = 1 - ratio
        two_hot_labels[np.arange(imgs.shape[0]), new_labels] += ratio

        return imgs, two_hot_labels
    else:
        two_hot_labels[np.arange(imgs.shape[0]), labels[0:imgs.shape[0]]] = 1.0
        return imgs, two_hot_labels

import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_PMix(nn.Module):
    def __init__(self, dataset, depth, num_classes, num_img, bottleneck=False):
        super(ResNet_PMix, self).__init__()        
        self.dataset = dataset
        self.tanh = nn.Tanh()
        if self.dataset.startswith('cifar'):
            self.inplanes = 16
            print(bottleneck)
            if bottleneck == True:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock

            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 16, n)
            self.layer2 = self._make_layer(block, 32, n, stride=2)
            self.layer3 = self._make_layer(block, 64, n, stride=2) 
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(64 * block.expansion * num_img, num_classes)
            

        elif dataset == 'imagenet':
            blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
            layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
            assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

            self.inplanes = 64
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
            self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
            self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
            self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2)
            self.avgpool = nn.AvgPool2d(7) 
            self.fc = nn.Linear((512 * blocks[depth].expansion * num_img), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = self.tanh(x)

        elif self.dataset == 'imagenet':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            
            x = self.fc(x)
            x = self.tanh(x)
    
        return x
    
    


class ATMModel(nn.Module):
    def __init__(self, num_classes):
        super(ATMModel, self).__init__()
        self.tanh = nn.Tanh()
        self.inplanes = 16
        #print(bottleneck)
        n = int((18 - 2) / 6)
        block = BasicBlock

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2) 
        self.avgpool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.tanh(x)
    
        return x
    

class ATMModel_tt(nn.Module):
    def __init__(self, num_classes):
        super(ATMModel_tt, self).__init__()
        self.tanh = nn.Tanh()
        self.inplanes = 16
        #print(bottleneck)
        n = int((18 - 2) / 6)
        block = BasicBlock

        self.conv1 = nn.Conv2d(4, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2) 
        self.avgpool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.tanh(x)
    
        return x
class ATMModel_ratio(nn.Module):
    def __init__(self, num_classes):
        super(ATMModel_ratio, self).__init__()
        self.tanh = nn.Tanh()
        self.inplanes = 16
        #print(bottleneck)
        n = int((18 - 2) / 6)
        block = BasicBlock

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2) 
        self.avgpool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.tanh(x)
    
        return x
    

class ATMModel_ratio_scale(nn.Module):
    def __init__(self, num_classes):
        super(ATMModel_ratio_scale, self).__init__()
        self.tanh = nn.Tanh()
        self.inplanes = 16
        #print(bottleneck)
        n = int((18 - 2) / 6)
        block = BasicBlock

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2) 
        self.avgpool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.tanh(x)
    
        return x

class ATMModel_dim(nn.Module):
    def __init__(self, num_classes):
        super(ATMModel_dim, self).__init__()
        self.tanh = nn.Tanh()
        self.inplanes = 16
        #print(bottleneck)
        n = int((18 - 2) / 6)
        block = BasicBlock

        self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2) 
        self.avgpool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.tanh(x)
    
        return x
    
class ATMModel_T(nn.Module):
    def __init__(self, num_classes):
        super(ATMModel_T, self).__init__()
        self.tanh = nn.Tanh()
        self.inplanes = 16
        #print(bottleneck)
        n = int((18 - 2) / 6)
        block = BasicBlock

        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2) 
        self.avgpool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(12288, 4096)
        self.fc2 = nn.Linear(4096, num_classes)
        
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.tanh(x)
    
        return x
    
class AtmModel_imagenet(nn.Module):
    def __init__(self, num_classes):
        super(AtmModel_imagenet, self).__init__()
        self.tanh = nn.Tanh()
        blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
        layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
        depth = 18
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
        self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
        self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
        self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2)
        self.avgpool = nn.AvgPool2d(3) 
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        
        
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.tanh(x)
    
        return x

def output_augment_ratio(trans, all_imgs1, all_imgs_f1, all_nor_imgs_f1, args):
    
    batch_size = all_imgs1.shape[1]
    
    trans = trans.cuda(args.gpu)
    
    all_imgs2 = torch.zeros(args.num_mixing,batch_size,3,args.image_size,args.image_size).cuda(args.gpu)
    all_imgs_f2 = torch.zeros(args.num_mixing,batch_size,1,args.image_size,args.image_size).cuda(args.gpu)
    all_nor_imgs_f2 = torch.zeros(args.num_mixing,batch_size,1,args.image_size,args.image_size).cuda(args.gpu)

    for num_mixing in range(args.num_mixing):
        
        imgs = all_imgs1[num_mixing].cuda(args.gpu)
        imgs_f = all_imgs_f1[num_mixing].cuda(args.gpu)
        img_f_nor = all_nor_imgs_f1[num_mixing].cuda(args.gpu)
        
        r_scale = 5
        t_scale = 8
        s_scale = 2
        
        n_p = 4
        
        rotation = trans[:,0+num_mixing*n_p] * r_scale #angle in degree, default:0 (-20 to 20)
        translate = trans[:,1+num_mixing*n_p:3+num_mixing*n_p] * args.image_size / t_scale  #pixel, default:0
        scale = trans[:,3+num_mixing*n_p] / s_scale + 1 #addtional scale, default:0

        scaleR = scale.clone()[:,None,None,None]
        scaleR[scaleR<1.0] = 1.0
        
        scaleXY = torch.reshape(scale,(batch_size,1)).repeat(1, 2).cuda(args.gpu)
        center = torch.zeros(batch_size,2).cuda(args.gpu)
        center[:] = (args.image_size-1)/2
        
        M = kornia.geometry.transform.get_affine_matrix2d(translate, center, scaleXY, rotation)
        M = M[:,:2,:]
        
        #imgs = kornia.geometry.transform.scale(imgs,scaleXY,center)
        #imgs = kornia.geometry.transform.rotate(imgs,rotation,center)
        #imgs = kornia.geometry.transform.translate(imgs,translate)
        imgs = kornia.geometry.transform.warp_affine(imgs, M, (args.image_size, args.image_size))
        #imgs = core.warp_perspective(imgs, M, (args.image_size, args.image_size))
        
        #imgs_f = kornia.geometry.transform.scale(imgs_f,scaleXY,center)
        #imgs_f = kornia.geometry.transform.rotate(imgs_f,rotation,center)
        #imgs_f = kornia.geometry.transform.translate(imgs_f,translate)
        imgs_f = kornia.geometry.transform.warp_affine(imgs_f, M, (args.image_size, args.image_size))
        #imgs_f = core.warp_perspective(imgs_f, M, (args.image_size, args.image_size))
        imgs_f = imgs_f/scaleR/scaleR
        
        #img_f_nor = kornia.geometry.transform.scale(img_f_nor,scaleXY,center)
        #img_f_nor = kornia.geometry.transform.rotate(img_f_nor,rotation,center)
        #img_f_nor = kornia.geometry.transform.translate(img_f_nor,translate)
        img_f_nor = kornia.geometry.transform.warp_affine(img_f_nor, M, (args.image_size, args.image_size))
        #img_f_nor = core.warp_perspective(img_f_nor, M, (args.image_size, args.image_size))
        img_f_nor = img_f_nor/scaleR/scaleR
        
        all_imgs2[num_mixing] = imgs
        all_imgs_f2[num_mixing] = imgs_f
        all_nor_imgs_f2[num_mixing] = img_f_nor
        

    xxx = (rotation[0],translate[0,0],translate[0,1],scale[0])
    
    return all_imgs2,all_imgs_f2,all_nor_imgs_f2,xxx

def output_augment_ratio_scale(trans, all_imgs1, all_imgs_f1, all_nor_imgs_f1, args):
    
    batch_size = all_imgs1.shape[1]
    
    trans = trans.cuda(args.gpu)
    
    all_imgs2 = torch.zeros(args.num_mixing,batch_size,3,args.image_size,args.image_size).cuda(args.gpu)
    all_imgs_f2 = torch.zeros(args.num_mixing,batch_size,1,args.image_size,args.image_size).cuda(args.gpu)
    all_nor_imgs_f2 = torch.zeros(args.num_mixing,batch_size,1,args.image_size,args.image_size).cuda(args.gpu)

    for num_mixing in range(args.num_mixing):
        
        imgs = all_imgs1[num_mixing].cuda(args.gpu)
        imgs_f = all_imgs_f1[num_mixing].cuda(args.gpu)
        img_f_nor = all_nor_imgs_f1[num_mixing].cuda(args.gpu)
        
        r_scale = 5
        t_scale = 8
        
        n_p = 3
        
        rotation = trans[:,0+num_mixing*n_p] * r_scale #angle in degree, default:0 (-20 to 20)
        translate = trans[:,1+num_mixing*n_p:3+num_mixing*n_p] * args.image_size / t_scale  #pixel, default:0
        scale = torch.ones(batch_size).cuda(args.gpu)
        #translate = trans[:,3+num_mixing*n_p] / s_scale + 1 #addtional scale, default:0

        scaleR = scale.clone()[:,None,None,None]
        scaleR[scaleR<1.0] = 1.0
        
        scaleXY = torch.reshape(scale,(batch_size,1)).repeat(1, 2).cuda(args.gpu)
        center = torch.zeros(batch_size,2).cuda(args.gpu)
        center[:] = (args.image_size-1)/2
        
        M = kornia.geometry.transform.get_affine_matrix2d(translate, center, scaleXY, rotation)
        M = M[:,:2,:]
        
        #imgs = kornia.geometry.transform.scale(imgs,scaleXY,center)
        #imgs = kornia.geometry.transform.rotate(imgs,rotation,center)
        #imgs = kornia.geometry.transform.translate(imgs,translate)
        imgs = kornia.geometry.transform.warp_affine(imgs, M, (args.image_size, args.image_size))
        #imgs = core.warp_perspective(imgs, M, (args.image_size, args.image_size))
        
        #imgs_f = kornia.geometry.transform.scale(imgs_f,scaleXY,center)
        #imgs_f = kornia.geometry.transform.rotate(imgs_f,rotation,center)
        #imgs_f = kornia.geometry.transform.translate(imgs_f,translate)
        imgs_f = kornia.geometry.transform.warp_affine(imgs_f, M, (args.image_size, args.image_size))
        #imgs_f = core.warp_perspective(imgs_f, M, (args.image_size, args.image_size))
        imgs_f = imgs_f/scaleR/scaleR
        
        #img_f_nor = kornia.geometry.transform.scale(img_f_nor,scaleXY,center)
        #img_f_nor = kornia.geometry.transform.rotate(img_f_nor,rotation,center)
        #img_f_nor = kornia.geometry.transform.translate(img_f_nor,translate)
        img_f_nor = kornia.geometry.transform.warp_affine(img_f_nor, M, (args.image_size, args.image_size))
        #img_f_nor = core.warp_perspective(img_f_nor, M, (args.image_size, args.image_size))
        img_f_nor = img_f_nor/scaleR/scaleR
        
        all_imgs2[num_mixing] = imgs
        all_imgs_f2[num_mixing] = imgs_f
        all_nor_imgs_f2[num_mixing] = img_f_nor
        

    xxx = (rotation[0],translate[0,0],translate[0,1],scale[0])
    
    return all_imgs2,all_imgs_f2,all_nor_imgs_f2,xxx


def output_augment_ratio_scale_2(trans, all_imgs1, all_imgs_f1, all_nor_imgs_f1, args):
    
    batch_size = all_imgs1.shape[1]
    
    trans = trans.cuda(args.gpu)
    
    all_imgs2 = torch.zeros(args.num_mixing,batch_size,3,args.image_size,args.image_size).cuda(args.gpu)
    all_imgs_f2 = torch.zeros(args.num_mixing,batch_size,1,args.image_size,args.image_size).cuda(args.gpu)
    all_nor_imgs_f2 = torch.zeros(args.num_mixing,batch_size,1,args.image_size,args.image_size).cuda(args.gpu)

    for num_mixing in range(args.num_mixing):
        
        imgs = all_imgs1[num_mixing].cuda(args.gpu)
        imgs_f = all_imgs_f1[num_mixing].cuda(args.gpu)
        img_f_nor = all_nor_imgs_f1[num_mixing].cuda(args.gpu)
        
        t_scale = 8
        
        n_p = 2
        
        # rotation = trans[:,0+num_mixing*n_p] * r_scale #angle in degree, default:0 (-20 to 20)
        rotation = torch.zeros(batch_size).cuda(args.gpu)
        translate = trans[:,0+num_mixing*n_p:2+num_mixing*n_p] * args.image_size / t_scale  #pixel, default:0
        scale = torch.ones(batch_size).cuda(args.gpu)
        #translate = trans[:,3+num_mixing*n_p] / s_scale + 1 #addtional scale, default:0

        scaleR = scale.clone()[:,None,None,None]
        scaleR[scaleR<1.0] = 1.0
        
        scaleXY = torch.reshape(scale,(batch_size,1)).repeat(1, 2).cuda(args.gpu)
        center = torch.zeros(batch_size,2).cuda(args.gpu)
        center[:] = (args.image_size-1)/2
        
        M = kornia.geometry.transform.get_affine_matrix2d(translate, center, scaleXY, rotation)
        M = M[:,:2,:]
        
        #imgs = kornia.geometry.transform.scale(imgs,scaleXY,center)
        #imgs = kornia.geometry.transform.rotate(imgs,rotation,center)
        #imgs = kornia.geometry.transform.translate(imgs,translate)
        imgs = kornia.geometry.transform.warp_affine(imgs, M, (args.image_size, args.image_size))
        #imgs = core.warp_perspective(imgs, M, (args.image_size, args.image_size))
        
        #imgs_f = kornia.geometry.transform.scale(imgs_f,scaleXY,center)
        #imgs_f = kornia.geometry.transform.rotate(imgs_f,rotation,center)
        #imgs_f = kornia.geometry.transform.translate(imgs_f,translate)
        imgs_f = kornia.geometry.transform.warp_affine(imgs_f, M, (args.image_size, args.image_size))
        #imgs_f = core.warp_perspective(imgs_f, M, (args.image_size, args.image_size))
        imgs_f = imgs_f/scaleR/scaleR
        
        #img_f_nor = kornia.geometry.transform.scale(img_f_nor,scaleXY,center)
        #img_f_nor = kornia.geometry.transform.rotate(img_f_nor,rotation,center)
        #img_f_nor = kornia.geometry.transform.translate(img_f_nor,translate)
        img_f_nor = kornia.geometry.transform.warp_affine(img_f_nor, M, (args.image_size, args.image_size))
        #img_f_nor = core.warp_perspective(img_f_nor, M, (args.image_size, args.image_size))
        img_f_nor = img_f_nor/scaleR/scaleR
        
        all_imgs2[num_mixing] = imgs
        all_imgs_f2[num_mixing] = imgs_f
        all_nor_imgs_f2[num_mixing] = img_f_nor
        

    xxx = (rotation[0],translate[0,0],translate[0,1],scale[0])
    
    return all_imgs2,all_imgs_f2,all_nor_imgs_f2,xxx


def output_augment_ratio_2(trans, all_imgs1, all_imgs_f1, all_nor_imgs_f1, args):
    
    batch_size = all_imgs1.shape[1]
    
    trans = trans.cuda(args.gpu)
    
    all_imgs2 = torch.zeros(args.num_mixing,batch_size,3,args.image_size,args.image_size).cuda(args.gpu)
    all_imgs_f2 = torch.zeros(args.num_mixing,batch_size,1,args.image_size,args.image_size).cuda(args.gpu)
    all_nor_imgs_f2 = torch.zeros(args.num_mixing,batch_size,1,args.image_size,args.image_size).cuda(args.gpu)

    for num_mixing in range(args.num_mixing):
        
        imgs = all_imgs1[num_mixing].cuda(args.gpu)
        imgs_f = all_imgs_f1[num_mixing].cuda(args.gpu)
        img_f_nor = all_nor_imgs_f1[num_mixing].cuda(args.gpu)
        
        t_scale = 8
        s_scale = 2
        
        n_p = 3
        
        # rotation = trans[:,0+num_mixing*n_p] * r_scale #angle in degree, default:0 (-20 to 20)
        rotation = torch.zeros(batch_size).cuda(args.gpu)
        translate = trans[:,0+num_mixing*n_p:2+num_mixing*n_p] * args.image_size / t_scale  #pixel, default:0
        # scale = torch.ones(batch_size).cuda(args.gpu)
        scale = trans[:,2+num_mixing*n_p] / s_scale + 1.0 #addtional scale, default:0
        #translate = trans[:,3+num_mixing*n_p] / s_scale + 1 #addtional scale, default:0

        scaleR = scale.clone()[:,None,None,None]
        scaleR[scaleR<1.0] = 1.0
        
        scaleXY = torch.reshape(scale,(batch_size,1)).repeat(1, 2).cuda(args.gpu)
        center = torch.zeros(batch_size,2).cuda(args.gpu)
        center[:] = (args.image_size-1)/2
        
        M = kornia.geometry.transform.get_affine_matrix2d(translate, center, scaleXY, rotation)
        M = M[:,:2,:]
        
        #imgs = kornia.geometry.transform.scale(imgs,scaleXY,center)
        #imgs = kornia.geometry.transform.rotate(imgs,rotation,center)
        #imgs = kornia.geometry.transform.translate(imgs,translate)
        imgs = kornia.geometry.transform.warp_affine(imgs, M, (args.image_size, args.image_size))
        #imgs = core.warp_perspective(imgs, M, (args.image_size, args.image_size))
        
        #imgs_f = kornia.geometry.transform.scale(imgs_f,scaleXY,center)
        #imgs_f = kornia.geometry.transform.rotate(imgs_f,rotation,center)
        #imgs_f = kornia.geometry.transform.translate(imgs_f,translate)
        imgs_f = kornia.geometry.transform.warp_affine(imgs_f, M, (args.image_size, args.image_size))
        #imgs_f = core.warp_perspective(imgs_f, M, (args.image_size, args.image_size))
        imgs_f = imgs_f/scaleR/scaleR
        
        #img_f_nor = kornia.geometry.transform.scale(img_f_nor,scaleXY,center)
        #img_f_nor = kornia.geometry.transform.rotate(img_f_nor,rotation,center)
        #img_f_nor = kornia.geometry.transform.translate(img_f_nor,translate)
        img_f_nor = kornia.geometry.transform.warp_affine(img_f_nor, M, (args.image_size, args.image_size))
        #img_f_nor = core.warp_perspective(img_f_nor, M, (args.image_size, args.image_size))
        img_f_nor = img_f_nor/scaleR/scaleR
        
        all_imgs2[num_mixing] = imgs
        all_imgs_f2[num_mixing] = imgs_f
        all_nor_imgs_f2[num_mixing] = img_f_nor
        

    xxx = (rotation[0],translate[0,0],translate[0,1],scale[0])
    
    return all_imgs2,all_imgs_f2,all_nor_imgs_f2,xxx

def output_augment(trans, all_imgs1, all_imgs_f1, all_nor_imgs_f1, args):
  
    batch_size = all_imgs1.shape[1]
    
    trans = trans.cuda(args.gpu)
    
    all_imgs2 = torch.zeros(args.num_mixing,batch_size,3,args.image_size,args.image_size).cuda(args.gpu)
    all_imgs_f2 = torch.zeros(args.num_mixing,batch_size,1,args.image_size,args.image_size).cuda(args.gpu)
    all_nor_imgs_f2 = torch.zeros(args.num_mixing,batch_size,1,args.image_size,args.image_size).cuda(args.gpu)

    for num_mixing in range(args.num_mixing):
        
        imgs = all_imgs1[num_mixing].cuda(args.gpu)
        imgs_f = all_imgs_f1[num_mixing].cuda(args.gpu)
        img_f_nor = all_nor_imgs_f1[num_mixing].cuda(args.gpu)
        
        t_scale = 8
        
        n_p = 2
        
        # rotation = trans[:,0+num_mixing*n_p] * r_scale #angle in degree, default:0 (-20 to 20)
        rotation = torch.zeros(batch_size).cuda(args.gpu)
        translate = trans[:,0+num_mixing*n_p:2+num_mixing*n_p] * args.image_size / t_scale  #pixel, default:0
        scale = torch.ones(batch_size).cuda(args.gpu)
        #translate = trans[:,3+num_mixing*n_p] / s_scale + 1 #addtional scale, default:0

        scaleR = scale.clone()[:,None,None,None]
        scaleR[scaleR<1.0] = 1.0
        
        scaleXY = torch.reshape(scale,(batch_size,1)).repeat(1, 2).cuda(args.gpu)
        center = torch.zeros(batch_size,2).cuda(args.gpu)
        center[:] = (args.image_size-1)/2
        
        
        M = kornia.geometry.transform.get_affine_matrix2d(translate, center, scaleXY, rotation)
        M = M[:,:2,:]
        
        #imgs = kornia.geometry.transform.scale(imgs,scaleXY,center)
        #imgs = kornia.geometry.transform.rotate(imgs,rotation,center)
        #imgs = kornia.geometry.transform.translate(imgs,translate)
        imgs = kornia.geometry.transform.warp_affine(imgs, M, (args.image_size, args.image_size))
        #imgs = core.warp_perspective(imgs, M, (args.image_size, args.image_size))
        
        #imgs_f = kornia.geometry.transform.scale(imgs_f,scaleXY,center)
        #imgs_f = kornia.geometry.transform.rotate(imgs_f,rotation,center)
        #imgs_f = kornia.geometry.transform.translate(imgs_f,translate)
        imgs_f = kornia.geometry.transform.warp_affine(imgs_f, M, (args.image_size, args.image_size))
        #imgs_f = core.warp_perspective(imgs_f, M, (args.image_size, args.image_size))
        imgs_f = imgs_f/scaleR/scaleR
        
        #img_f_nor = kornia.geometry.transform.scale(img_f_nor,scaleXY,center)
        #img_f_nor = kornia.geometry.transform.rotate(img_f_nor,rotation,center)
        #img_f_nor = kornia.geometry.transform.translate(img_f_nor,translate)
        img_f_nor = kornia.geometry.transform.warp_affine(img_f_nor, M, (args.image_size, args.image_size))
        #img_f_nor = core.warp_perspective(img_f_nor, M, (args.image_size, args.image_size))
        img_f_nor = img_f_nor/scaleR/scaleR
        
        all_imgs2[num_mixing] = imgs
        all_imgs_f2[num_mixing] = imgs_f
        all_nor_imgs_f2[num_mixing] = img_f_nor
        

    xxx = (rotation[0],translate[0,0],translate[0,1],scale[0])
    
    return all_imgs2,all_imgs_f2,all_nor_imgs_f2,xxx

def scale_ops(all_imgs0, all_imgs_f0, args):
    
    batch_size = all_imgs0.shape[1]
    combined = torch.cat((all_imgs0[0], all_imgs_f0[0]), 1)

    a = 0.5
    b = 1.5
    scale_rate = (b - a) * torch.rand(batch_size) + a
    scale_rate = scale_rate[:,None].repeat(1, 2).cuda(args.gpu)
    
    center = torch.randint(0, args.image_size, (batch_size,2)).cuda(args.gpu, non_blocking=True)
    
    combined = kornia.geometry.transform.scale(combined, scale_rate, center=center.float())
    
    all_imgs0[0] = combined[:,:3,:,:]
    all_imgs_f0[0] = torch.unsqueeze(combined[:,3,:,:], 1)
    
    return all_imgs0, all_imgs_f0

def scale_ops_2(all_imgs0, all_imgs_f0, all_nor_imgs_f0, args):
    
    batch_size = all_imgs0.shape[1]
    combined = torch.cat((all_imgs0[0], all_imgs_f0[0], all_nor_imgs_f0[0]), 1)

    a = 0.5
    b = 1.0
    # a = 0.25
    # b = 1.25
    scale_rate_raw = ((b - a) * torch.rand(batch_size) + a).cuda(args.gpu)
    scale_rate = scale_rate_raw[:,None].repeat(1, 2)
    
    center = torch.randint(0, args.image_size, (batch_size,2)).cuda(args.gpu, non_blocking=True)
    
    combined = kornia.geometry.transform.scale(combined, scale_rate, center=center.float())
    
    all_imgs0[0] = combined[:,:3,:,:]
    # all_imgs_f0[0] = torch.unsqueeze(combined[:,3,:,:], 1)
    
    #print(sc) * scale_rate_raw[:,None,None]*scale_rate_raw[:,None,None]
    all_imgs_f0[0] = torch.unsqueeze(combined[:,3,:,:], 1)
    all_nor_imgs_f0[0] = torch.unsqueeze(combined[:,4,:,:], 1)

    return all_imgs0, all_imgs_f0, all_nor_imgs_f0
    
def output_augment_scale(trans, all_imgs1, all_imgs_f1, all_nor_imgs_f1, args):
    
    batch_size = all_imgs1.shape[1]
    
    trans = trans.cuda(args.gpu)
    
    all_imgs2 = torch.zeros(args.num_mixing,batch_size,3,args.image_size,args.image_size).cuda(args.gpu)
    all_imgs_f2 = torch.zeros(args.num_mixing,batch_size,1,args.image_size,args.image_size).cuda(args.gpu)
    all_nor_imgs_f2 = torch.zeros(args.num_mixing,batch_size,1,args.image_size,args.image_size).cuda(args.gpu)

    for num_mixing in range(args.num_mixing):
        
        imgs = all_imgs1[num_mixing].cuda(args.gpu)
        imgs_f = all_imgs_f1[num_mixing].cuda(args.gpu)
        img_f_nor = all_nor_imgs_f1[num_mixing].cuda(args.gpu)
        
        # r_scale = 0
        t_scale = 8
        # s_scale = 1e6
        
        n_p = 2
        
        translate = trans[:,0+num_mixing*n_p:2+num_mixing*n_p] * args.image_size / t_scale  #pixel, default:0
        
        #rotation = trans[:,0+num_mixing*n_p] * r_scale #angle in degree, default:0 (-20 to 20)
        #translate = trans[:,1+num_mixing*n_p:3+num_mixing*n_p] * args.image_size / t_scale  #pixel, default:0
        #scale = trans[:,3+num_mixing*n_p] / s_scale + 1 #addtional scale, default:0

        scale = torch.ones(batch_size).cuda(args.gpu)
        rotation = torch.zeros(batch_size).cuda(args.gpu)
        
        scaleR = scale.clone()[:,None,None,None]
        scaleR[scaleR<1.0] = 1.0
        
        scaleXY = torch.reshape(scale,(batch_size,1)).repeat(1, 2).cuda(args.gpu)
        center = torch.zeros(batch_size,2).cuda(args.gpu)
        center[:] = (args.image_size-1)/2
        
        M = kornia.geometry.transform.get_affine_matrix2d(translate, center, scaleXY, rotation)
        M = M[:,:2,:]
        
        #imgs = kornia.geometry.transform.scale(imgs,scaleXY,center)
        #imgs = kornia.geometry.transform.rotate(imgs,rotation,center)
        #imgs = kornia.geometry.transform.translate(imgs,translate)
        imgs = kornia.geometry.transform.warp_affine(imgs, M, (args.image_size, args.image_size))
        #imgs = core.warp_perspective(imgs, M, (args.image_size, args.image_size))
        
        #imgs_f = kornia.geometry.transform.scale(imgs_f,scaleXY,center)
        #imgs_f = kornia.geometry.transform.rotate(imgs_f,rotation,center)
        #imgs_f = kornia.geometry.transform.translate(imgs_f,translate)
        imgs_f = kornia.geometry.transform.warp_affine(imgs_f, M, (args.image_size, args.image_size))
        #imgs_f = core.warp_perspective(imgs_f, M, (args.image_size, args.image_size))
        imgs_f = imgs_f/scaleR/scaleR
        
        #img_f_nor = kornia.geometry.transform.scale(img_f_nor,scaleXY,center)
        #img_f_nor = kornia.geometry.transform.rotate(img_f_nor,rotation,center)
        #img_f_nor = kornia.geometry.transform.translate(img_f_nor,translate)
        img_f_nor = kornia.geometry.transform.warp_affine(img_f_nor, M, (args.image_size, args.image_size))
        #img_f_nor = core.warp_perspective(img_f_nor, M, (args.image_size, args.image_size))
        img_f_nor = img_f_nor/scaleR/scaleR
        
        all_imgs2[num_mixing] = imgs
        all_imgs_f2[num_mixing] = imgs_f
        all_nor_imgs_f2[num_mixing] = img_f_nor
        

    xxx = (rotation[0],translate[0,0],translate[0,1],scale[0])
    
    return all_imgs2,all_imgs_f2,all_nor_imgs_f2,xxx

from randaugment import RandAugment

ra = transforms.Compose([
])

# Add RandAugment with N, M(hyperparameter)
ra.transforms.insert(0, RandAugment(2, 14))

class ATMB(nn.Module):
    def __init__(self, mask_pool_size):
        super(ATMB, self).__init__()
        
        # # Automated scale module
        # if args.enable_auto_scale:
        # self.convS1 = nn.Conv2d(3, 8, kernel_size=3, stride=3, padding=0)
        # self.convS2 = nn.Conv2d(8, 16, kernel_size=3, stride=3, padding=0)
        # self.fcS1 = nn.Linear(144, 72)
        # self.fcS2 = nn.Linear(72, 4)
        
        # Automated translate module
        self.convF1 = nn.Conv2d(3, 8, kernel_size=3, stride=3, padding=0)
        self.convF2 = nn.Conv2d(8, 16, kernel_size=3, stride=3, padding=0)
        self.fcF1 = nn.Linear(144, 72)
        self.fcF2 = nn.Linear(72, 2)
        self.tanh = nn.Tanh()

        # Automated mask module
        # self.convt1 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=0)
        self.convt1 = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.avgpool2d = nn.AvgPool2d(mask_pool_size)
        self.upsample = nn.Upsample(32)
        
        
    def forward(self, I1, I2, F1, F2, lam, args):
        
        F1R = F1.clone()
        F2R = F2.clone()
        
        # IF2 = torch.cat((I2,F2),1)
        # IF2 = ra(IF2)
        # I2 = IF2[:,:3,:,:]
        # F2 = IF2[:,3,:,:]
        
        # F2 = torch.unsqueeze(F2,1)
        
        # Random rotate module
        if args.enable_random_rotate:
            a = -args.rotate_angle
            b = args.rotate_angle
            rotate_rate = ((b - a) * torch.rand(I1.shape[0]) + a).cuda()
            center = (torch.rand((I1.shape[0],2))*I1.shape[2]).cuda()
            I2 = kornia.geometry.transform.rotate(I2, rotate_rate, center=center)
            F2 = kornia.geometry.transform.rotate(F2, rotate_rate, center=center)
        
        # Random scale module
        if args.enable_random_scale:
            
            a = args.lower #0.5
            b = args.upper #1.2
            scale_rate_raw = ((b - a) * torch.rand(I1.shape[0]) + a).cuda()
                
        if args.enable_lam_scale:
            
            lam_ratio = (1-lam[:,0,0,0])/2 +0.5
            a = -args.lam_scale_margin
            b = args.lam_scale_margin
            margin = ((b - a) * torch.rand(I1.shape[0]) + a).cuda()
            scale_rate_raw = lam_ratio + margin
        
        if args.enable_lam_scale or args.enable_random_scale:
            scale_rate = scale_rate_raw[:,None].repeat(1, 2)
            center = (torch.rand((I1.shape[0],2))*I1.shape[2]).cuda()
            I2 = kornia.geometry.transform.scale(I2, scale_rate, center=center)
            F2 = kornia.geometry.transform.scale(F2, scale_rate, center=center)
            
        # # Automated scale module
        # if args.enable_auto_scale:
        #     F1P2L = torch.cat((F1,F2,(1-lam)), 1)
        #     # F1P2L = torch.cat((F1,F2,lam), 1)
        #     F1P2L = self.convS1(F1P2L)
        #     F1P2L = self.convS2(F1P2L)
        #     F1P2L = F1P2L.view(F1P2L.size(0), -1)
        #     F1P2L = self.fcS1(F1P2L)
        #     F1P2L = self.fcS2(F1P2L)
        #     S2 = self.sigmoid(F1P2L)
        #     SS2 = S2[:,:2]
        #     SC2 = S2[:,2:]*I1.shape[2]
        #     F2 = kornia.geometry.transform.scale(F2, SS2, SC2)
        #     I2 = kornia.geometry.transform.scale(I2, SS2, SC2)
        
        # Automated translate module
        if args.enable_auto_translate:
            coe_trans = (I1.shape[2] / args.trans_scale)
            # F12 = F1*F2
            # F2S = torch.cat((F2,F12,lam), 1)
            F2S = torch.cat((F1,F2,lam), 1)
            F2S = self.convF1(F2S)
            F2S = self.convF2(F2S)
            F2S = F2S.view(F2S.size(0), -1)
            F2S = self.fcF1(F2S)
            F2S = self.fcF2(F2S)
            T2 = self.tanh(F2S)
            T2 = T2 * coe_trans
            F2 = kornia.geometry.transform.translate(F2, T2)
            I2 = kornia.geometry.transform.translate(I2, T2)
        else:
            T2 = None
        
        # Automated mask module
        # F1N2 = F1-F2
        # F1N2L = torch.cat((F1, F2, F1N2, lam), 1)
        F1N2L = torch.cat((F1, F2, lam), 1)
        mask = self.convt1(F1N2L)
        mask = self.sigmoid(mask)
        mask = self.avgpool2d(mask)
        mask = self.upsample(mask)
        mixed_image = I1 * mask + I2 * (1-mask)
        
        #Generate semantic label ratio for feedback
        R1 = F1 * mask
        R2 = F2 * (1-mask)
        
        #get value of deleted information
        di1 = torch.sum(F1R,(1,2,3))-torch.sum(R1,(1,2,3))
        di2 = torch.sum(F2R,(1,2,3))-torch.sum(R2,(1,2,3))
        
        #Collect data for understanding
        if args.show_data:
            if T2 is not None:
                d = mask, T2, di1, di2
            else:
                d = mask, di1, di2
            return mixed_image, R1, R2, d
        
        return mixed_image, R1, R2#, data


class ATMB_old(nn.Module):
    def __init__(self):
        super(ATMB, self).__init__()
        
        # self.conv1F1 = nn.Conv2d(2, 8, kernel_size=3, stride=3, padding=0)
        # self.conv1F2 = nn.Conv2d(8, 16, kernel_size=3, stride=3, padding=0)
        # # self.conv1F3 = nn.Conv2d(16, 32, kernel_size=3, stride=3, padding=0)
        # self.fc1F1 = nn.Linear(144, 72)
        # self.fc1F2 = nn.Linear(72, 2)
        
        self.convF1 = nn.Conv2d(2, 8, kernel_size=3, stride=3, padding=0)
        self.convF2 = nn.Conv2d(8, 16, kernel_size=3, stride=3, padding=0)
        # self.conv2F3 = nn.Conv2d(16, 32, kernel_size=3, stride=3, padding=0)
        self.fcF1 = nn.Linear(144, 72)
        self.fcF2 = nn.Linear(72, 2)
        
        # self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        # self.fc1 = nn.Linear(32, 2)
        
        # self.convp = nn.Conv2d(2, 10, kernel_size=1, stride=1, padding=0)
        # self.convz = nn.Conv2d(2, 10, kernel_size=1, stride=1, padding=0)
        self.convt1 = nn.Conv2d(4, 1, kernel_size=5, stride=1, padding=0)
        # self.convt2 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.avgpool2d = nn.AvgPool2d(28)
        self.upsample = nn.Upsample(32)
        
    def forward(self, I1, I2, F1, F2, lam, args):
        # coe_trans = (I1.shape[2] / 4)
        coe_trans = (I1.shape[2] / 2)
        
        if True:
            
            if False:
                lab_ratio = torch.sqrt(1-lam[:,0,0,0]/2)
                a = -0.1
                b = 0.1
                margin = ((b - a) * torch.rand(I1.shape[0]) + a).cuda()
                scale_rate_raw = lab_ratio + margin
            else:
                a = args.lower #0.5
                b = args.upper #1.2
                scale_rate_raw = ((b - a) * torch.rand(I1.shape[0]) + a).cuda()
            
            scale_rate = scale_rate_raw[:,None].repeat(1, 2)
            
            center = (torch.rand((I1.shape[0],2))*I1.shape[2]).cuda()
            
            I2 = kornia.geometry.transform.scale(I2, scale_rate, center=center)
            F2 = kornia.geometry.transform.scale(F2, scale_rate, center=center)
        
        F12 = F1*F2
        
        if True:
            # F1S = torch.cat((F1,F12), 1)
            # F1S = self.convF1(F1S)
            # F1S = self.convF2(F1S)
            # F1S = F1S.view(F1S.size(0), -1)
            # F1S = self.fcF1(F1S)
            # F1S = self.fcF2(F1S)
            # T1 = self.tanh(F1S)
            # T1 = T1 * coe_trans
            
            F2S = torch.cat((F2,F12), 1)
            F2S = self.convF1(F2S)
            F2S = self.convF2(F2S)
            F2S = F2S.view(F2S.size(0), -1)
            F2S = self.fcF1(F2S)
            F2S = self.fcF2(F2S)
            T2 = self.tanh(F2S)
            T2 = T2 * coe_trans
        
        else:
            F1S = torch.cat((F1,F12), 1)
            F1S = self.conv1(F1S)
            F1S = self.conv2(F1S)
            F1S = self.avgpool2d(F1S)
            F1S = F1S.view(F1S.size(0), -1)
            F1S = self.fc1(F1S)
            T1 = self.tanh(F1S)
            T1 = T1 * coe_trans
            
            F2S = torch.cat((F2,F12), 1)
            F2S = self.conv1(F2S)
            F2S = self.conv2(F2S)
            F2S = self.avgpool2d(F2S)
            F2S = F2S.view(F2S.size(0), -1)
            F2S = self.fc1(F2S)
            T2 = self.tanh(F2S)
            T2 = T2 * coe_trans
        
        if False:
            TF1 = kornia.geometry.transform.translate(F1, T1)
            TF1L = torch.cat((TF1,lam), 1)
            TF1L = self.avgpool2d(TF1L)
            TF1WP = self.convp(TF1L)
            
            TF1Z = self.convz(TF1L)
            
            TF2 = kornia.geometry.transform.translate(F2, T2)
            TF2L = torch.cat((TF2,(1-lam)), 1)
            TF2L = self.avgpool2d(TF2L)
            TF2WP = self.convp(TF2L)
            
            p = torch.matmul(TF1WP, TF2WP)
            p = self.softmax(p)
            mask = torch.matmul(TF1Z,p)
            mask = self.sigmoid(mask)
            mask = self.upsample(mask)
        
        else:
            # TF1 = kornia.geometry.transform.translate(F1, T1)
            TF1 = F1
            TF2 = kornia.geometry.transform.translate(F2, T2)
            TF1N2 = TF1-TF2
            TF1N2L = torch.cat((TF1, TF2, TF1N2, lam), 1)
            mask = self.convt1(TF1N2L)
            mask = self.sigmoid(mask)
            mask = self.upsample(mask)
        
        # TI1 = kornia.geometry.transform.translate(I1, T1)
        TI1 = I1
        TI2 = kornia.geometry.transform.translate(I2, T2)
        mixed_image = TI1 * mask + TI2 * (1-mask)
        
        R1 = TF1 * mask
        R2 = TF2 * (1-mask)
        
        T1 = torch.tensor([0,0])
        return mixed_image, R1, R2, mask, T1, T2, scale_rate_raw

class MB_M(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(MB_M, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, stride=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(size=(32,32))
        
    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        x = self.sigmoid(x)
        x = self.upsample(x)
    
        return x
    

class MB_T(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(MB_T, self).__init__()
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=5, stride=3, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=3, padding=0)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        #self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        x = self.tanh(x)
    
        return x

us = nn.Upsample(size=(32,32))
def ATM(imgs, img_feature, labels, args):
    two_hot_labels = torch.zeros((imgs.shape[0], args.num_class)).cuda(args.gpu)

    if torch.rand(1) <= args.prob:

        bs = imgs.shape[0]
        
        rand_index = torch.randperm(imgs.size()[0])
        all_imgs0 = torch.zeros(args.num_mixing,bs,3,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_l = torch.zeros(args.num_mixing,bs).cuda(args.gpu)
        
        all_imgs0[0] = imgs.clone()
        all_imgs_f0[0] = img_feature.clone()
        all_imgs_l[0] = labels.clone()
        
        for image_index in range(args.num_mixing-1) :
            rand_index = torch.randperm(imgs.size()[0])
            all_imgs0[image_index+1] = imgs[rand_index].clone()
            all_imgs_f0[image_index+1] = img_feature[rand_index].clone()
            all_imgs_l[image_index+1] = labels[rand_index].clone()
            
        all_nor_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        if torch.isnan(all_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
        all_nor_imgs_f0[0] = (all_imgs_f0[0] + 1e-6) / (torch.sum(all_imgs_f0[0], dim=(1,2,3))[:, None, None, None] + 1e-6)
        all_nor_imgs_f0[1] = (all_imgs_f0[1] + 1e-6) / (torch.sum(all_imgs_f0[1], dim=(1,2,3))[:, None, None, None] + 1e-6)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
            
        all_imgs_l = all_imgs_l.type(torch.LongTensor)

        inputs1 = all_imgs_f0[0].clone()
        for image_index in range(args.num_mixing-1):
            inputs1 =  torch.cat((inputs1, all_imgs_f0[image_index+1].clone()), 1)
            
        bt = torch.distributions.beta.Beta(1.0,1.0)
        
        input_ratio_raw = bt.sample(torch.Size([bs])).cuda(args.gpu)
        
        input_ratio = input_ratio_raw[:,None,None,None]
        input_ratio = input_ratio.repeat(1,1,args.image_size,args.image_size)
        inputs1 =  torch.cat((inputs1, input_ratio), 1)
        mixing_matrix = args.ATMModel_m(inputs1)

        # inputs2 =  torch.cat((inputs1, mixing_matrix), 1)
        # trans = args.ATMModel_t(inputs2)

        # all_imgs0,all_imgs_f0,all_nor_imgs_f0,xxx = output_augment(trans, all_imgs0, all_imgs_f0, all_nor_imgs_f0, args) 
    
        mixed_imgs = all_imgs0[0]*mixing_matrix + all_imgs0[1]*(1-mixing_matrix)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f00')
            
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f01')
            
        if torch.isnan(mixing_matrix).any():
            print('mixing_matrix')

        ratio1 = torch.sum(all_nor_imgs_f0[0]*mixing_matrix, (1,2,3)).cuda(args.gpu)
        ratio2 = torch.sum(all_nor_imgs_f0[1]*(1-mixing_matrix), (1,2,3)).cuda(args.gpu)
        
        if torch.isnan(ratio1).any():
            print('ratio1')
        if torch.isnan(ratio2).any():
            print('ratio2')

        total_info = (ratio1 + ratio2).cuda(args.gpu)

        ratio1 = ratio1/total_info
        ratio2 = ratio2/total_info
        #print(ratio1+ratio2)
        #print(ratio1.get_device(), two_hot_labels.get_device())
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[0]] += ratio1
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[1]] += ratio2

        return mixed_imgs, two_hot_labels
    else:
        two_hot_labels[np.arange(imgs.shape[0]), labels[0:imgs.shape[0]]] = 1.0
        return imgs, two_hot_labels
    
def ATM_dim(imgs, img_feature, labels, args):
    two_hot_labels = torch.zeros((imgs.shape[0], args.num_class)).cuda(args.gpu)

    if torch.rand(1) <= args.prob:

        bs = imgs.shape[0]
        
        rand_index = torch.randperm(imgs.size()[0])
        all_imgs0 = torch.zeros(args.num_mixing,bs,3,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_l = torch.zeros(args.num_mixing,bs).cuda(args.gpu)
        
        all_imgs0[0] = imgs.clone()
        all_imgs_f0[0] = img_feature.clone()
        all_imgs_l[0] = labels.clone()
        
        for image_index in range(args.num_mixing-1) :
            rand_index = torch.randperm(imgs.size()[0])
            all_imgs0[image_index+1] = imgs[rand_index].clone()
            all_imgs_f0[image_index+1] = img_feature[rand_index].clone()
            all_imgs_l[image_index+1] = labels[rand_index].clone()
            
        all_nor_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        if torch.isnan(all_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
        all_nor_imgs_f0[0] = (all_imgs_f0[0] + 1e-6) / (torch.sum(all_imgs_f0[0], dim=(1,2,3))[:, None, None, None] + 1e-6)
        all_nor_imgs_f0[1] = (all_imgs_f0[1] + 1e-6) / (torch.sum(all_imgs_f0[1], dim=(1,2,3))[:, None, None, None] + 1e-6)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
            
        all_imgs_l = all_imgs_l.type(torch.LongTensor)

        inputs1 = all_imgs_f0[0].clone()
        for image_index in range(args.num_mixing-1):
            inputs1 =  torch.cat((inputs1, all_imgs_f0[image_index+1].clone()), 1)
            
        output = args.ATMModel(inputs1)
        mixing_matrix = output[:,:args.mixing_sectors*args.mixing_sectors]
        trans = output[:,args.mixing_sectors*args.mixing_sectors:]

        mixing_matrix = torch.reshape(mixing_matrix, (bs,1, args.mixing_sectors, args.mixing_sectors))
        mixing_matrix = (mixing_matrix + 1) / 2
        mixing_matrix = transforms.functional.resize(mixing_matrix, args.image_size).clone()
        
        all_imgs0,all_imgs_f0,all_nor_imgs_f0,xxx = output_augment(trans, all_imgs0, all_imgs_f0, all_nor_imgs_f0, args)  
        
        mixed_imgs = all_imgs0[0]*mixing_matrix + all_imgs0[1]*(1-mixing_matrix)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f00')
            
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f01')
            
        if torch.isnan(mixing_matrix).any():
            print('mixing_matrix')

        ratio1 = torch.sum(all_nor_imgs_f0[0]*mixing_matrix, (1,2,3)).cuda(args.gpu)
        ratio2 = torch.sum(all_nor_imgs_f0[1]*(1-mixing_matrix), (1,2,3)).cuda(args.gpu)
        
        if torch.isnan(ratio1).any():
            print('ratio1')
        if torch.isnan(ratio2).any():
            print('ratio2')

        total_info = (ratio1 + ratio2).cuda(args.gpu)

        ratio1 = ratio1/total_info
        ratio2 = ratio2/total_info
        #print(ratio1+ratio2)
        #print(ratio1.get_device(), two_hot_labels.get_device())
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[0]] += ratio1
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[1]] += ratio2

        return mixed_imgs, two_hot_labels
    else:
        two_hot_labels[np.arange(imgs.shape[0]), labels[0:imgs.shape[0]]] = 1.0
        return imgs, two_hot_labels
    
def ATM_ratio(imgs, img_feature, labels, args):
    two_hot_labels = torch.zeros((imgs.shape[0], args.num_class)).cuda(args.gpu)

    if torch.rand(1) <= args.prob:

        bs = imgs.shape[0]
        
        rand_index = torch.randperm(imgs.size()[0])
        all_imgs0 = torch.zeros(args.num_mixing,bs,3,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_l = torch.zeros(args.num_mixing,bs).cuda(args.gpu)
        
        all_imgs0[0] = imgs.clone()
        all_imgs_f0[0] = img_feature.clone()
        all_imgs_l[0] = labels.clone()
        
        for image_index in range(args.num_mixing-1) :
            rand_index = torch.randperm(imgs.size()[0])
            all_imgs0[image_index+1] = imgs[rand_index].clone()
            all_imgs_f0[image_index+1] = img_feature[rand_index].clone()
            all_imgs_l[image_index+1] = labels[rand_index].clone()
            
        all_nor_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        if torch.isnan(all_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
        all_nor_imgs_f0[0] = (all_imgs_f0[0] + 1e-6) / (torch.sum(all_imgs_f0[0], dim=(1,2,3))[:, None, None, None] + 1e-6)
        all_nor_imgs_f0[1] = (all_imgs_f0[1] + 1e-6) / (torch.sum(all_imgs_f0[1], dim=(1,2,3))[:, None, None, None] + 1e-6)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
            
        all_imgs_l = all_imgs_l.type(torch.LongTensor)

        inputs1 = all_imgs_f0[0].clone()
        for image_index in range(args.num_mixing-1):
            inputs1 =  torch.cat((inputs1, all_imgs_f0[image_index+1].clone()), 1)
            
        bt = torch.distributions.beta.Beta(1.0,1.0)
        
        input_ratio_raw = bt.sample(torch.Size([bs])).cuda(args.gpu)
        
        # a = 0.1
        # b = 0.9
        # input_ratio_raw = ((b - a) * torch.rand(bs) + a).cuda(args.gpu)
        
        input_ratio = input_ratio_raw[:,None,None,None]
        input_ratio = input_ratio.repeat(1,1,args.image_size,args.image_size)
        inputs1 =  torch.cat((inputs1, input_ratio), 1)
        
        output = args.ATMModel(inputs1)
        mixing_matrix = output[:,:args.mixing_sectors*args.mixing_sectors]
        trans = output[:,args.mixing_sectors*args.mixing_sectors:]

        mixing_matrix = torch.reshape(mixing_matrix, (bs,1, args.mixing_sectors, args.mixing_sectors))
        mixing_matrix = (mixing_matrix + 1) / 2
        mixing_matrix = transforms.functional.resize(mixing_matrix, args.image_size).clone()
        
        all_imgs0,all_imgs_f0,all_nor_imgs_f0,xxx = output_augment_ratio(trans, all_imgs0, all_imgs_f0, all_nor_imgs_f0, args)  
        
        mixed_imgs = all_imgs0[0]*mixing_matrix + all_imgs0[1]*(1-mixing_matrix)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f00')
            
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f01')
            
        if torch.isnan(mixing_matrix).any():
            print('mixing_matrix')

        ratio1 = torch.sum(all_nor_imgs_f0[0]*mixing_matrix, (1,2,3)).cuda(args.gpu)
        ratio2 = torch.sum(all_nor_imgs_f0[1]*(1-mixing_matrix), (1,2,3)).cuda(args.gpu)
        
        if torch.isnan(ratio1).any():
            print('ratio1')
        if torch.isnan(ratio2).any():
            print('ratio2')

        total_info = (ratio1 + ratio2).cuda(args.gpu)

        ratio1 = ratio1/total_info
        ratio2 = ratio2/total_info
        #print(ratio1+ratio2)
        #print(ratio1.get_device(), two_hot_labels.get_device())
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[0]] += ratio1
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[1]] += ratio2

        return mixed_imgs, two_hot_labels
    else:
        two_hot_labels[np.arange(imgs.shape[0]), labels[0:imgs.shape[0]]] = 1.0
        return imgs, two_hot_labels
    

def ATM_ratio_scale(imgs, img_feature, labels, args):
    two_hot_labels = torch.zeros((imgs.shape[0], args.num_class)).cuda(args.gpu)

    if torch.rand(1) <= args.prob:

        bs = imgs.shape[0]
        
        rand_index = torch.randperm(imgs.size()[0])
        all_imgs0 = torch.zeros(args.num_mixing,bs,3,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_l = torch.zeros(args.num_mixing,bs).cuda(args.gpu)
        
        all_imgs0[0] = imgs.clone()
        all_imgs_f0[0] = img_feature.clone()
        all_imgs_l[0] = labels.clone()
        
        for image_index in range(args.num_mixing-1) :
            rand_index = torch.randperm(imgs.size()[0])
            all_imgs0[image_index+1] = imgs[rand_index].clone()
            all_imgs_f0[image_index+1] = img_feature[rand_index].clone()
            all_imgs_l[image_index+1] = labels[rand_index].clone()
            
        all_imgs0, all_imgs_f0 = scale_ops(all_imgs0, all_imgs_f0, args)
            
        all_nor_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        if torch.isnan(all_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
        all_nor_imgs_f0[0] = (all_imgs_f0[0] + 1e-6) / (torch.sum(all_imgs_f0[0], dim=(1,2,3))[:, None, None, None] + 1e-6)
        all_nor_imgs_f0[1] = (all_imgs_f0[1] + 1e-6) / (torch.sum(all_imgs_f0[1], dim=(1,2,3))[:, None, None, None] + 1e-6)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
            
        all_imgs_l = all_imgs_l.type(torch.LongTensor)

        inputs1 = all_imgs_f0[0].clone()
        for image_index in range(args.num_mixing-1):
            inputs1 =  torch.cat((inputs1, all_imgs_f0[image_index+1].clone()), 1)
            
        bt = torch.distributions.beta.Beta(1.0,1.0)
        
        input_ratio_raw = bt.sample(torch.Size([bs])).cuda(args.gpu)
        
        # a = 0.1
        # b = 0.9
        # input_ratio_raw = ((b - a) * torch.rand(bs) + a).cuda(args.gpu)
        
        input_ratio = input_ratio_raw[:,None,None,None]
        input_ratio = input_ratio.repeat(1,1,args.image_size,args.image_size)
        inputs1 =  torch.cat((inputs1, input_ratio), 1)
        
        output = args.ATMModel(inputs1)
        mixing_matrix = output[:,:args.mixing_sectors*args.mixing_sectors]
        trans = output[:,args.mixing_sectors*args.mixing_sectors:]

        mixing_matrix = torch.reshape(mixing_matrix, (bs,1, args.mixing_sectors, args.mixing_sectors))
        mixing_matrix = (mixing_matrix + 1) / 2
        mixing_matrix = transforms.functional.resize(mixing_matrix, args.image_size).clone()
        
        all_imgs0,all_imgs_f0,all_nor_imgs_f0,xxx = output_augment_ratio_scale(trans, all_imgs0, all_imgs_f0, all_nor_imgs_f0, args)  
        
        mixed_imgs = all_imgs0[0]*mixing_matrix + all_imgs0[1]*(1-mixing_matrix)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f00')
            
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f01')
            
        if torch.isnan(mixing_matrix).any():
            print('mixing_matrix')

        ratio1 = torch.sum(all_nor_imgs_f0[0]*mixing_matrix, (1,2,3)).cuda(args.gpu)
        ratio2 = torch.sum(all_nor_imgs_f0[1]*(1-mixing_matrix), (1,2,3)).cuda(args.gpu)
        
        if torch.isnan(ratio1).any():
            print('ratio1')
        if torch.isnan(ratio2).any():
            print('ratio2')

        total_info = (ratio1 + ratio2).cuda(args.gpu)

        ratio1 = ratio1/total_info
        ratio2 = ratio2/total_info
        #print(ratio1+ratio2)
        #print(ratio1.get_device(), two_hot_labels.get_device())
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[0]] += ratio1
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[1]] += ratio2

        return mixed_imgs, two_hot_labels
    else:
        two_hot_labels[np.arange(imgs.shape[0]), labels[0:imgs.shape[0]]] = 1.0
        return imgs, two_hot_labels


def ATM_ratio_scale_2(imgs, img_feature, labels, args):
    two_hot_labels = torch.zeros((imgs.shape[0], args.num_class)).cuda(args.gpu)

    if torch.rand(1) <= args.prob:

        bs = imgs.shape[0]
        
        rand_index = torch.randperm(imgs.size()[0])
        all_imgs0 = torch.zeros(args.num_mixing,bs,3,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_l = torch.zeros(args.num_mixing,bs).cuda(args.gpu)
        
        all_imgs0[0] = imgs.clone()
        all_imgs_f0[0] = img_feature.clone()
        all_imgs_l[0] = labels.clone()
        
        for image_index in range(args.num_mixing-1) :
            rand_index = torch.randperm(imgs.size()[0])
            all_imgs0[image_index+1] = imgs[rand_index].clone()
            all_imgs_f0[image_index+1] = img_feature[rand_index].clone()
            all_imgs_l[image_index+1] = labels[rand_index].clone()
            
        
            
        all_nor_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        if torch.isnan(all_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
        all_nor_imgs_f0[0] = (all_imgs_f0[0] + 1e-6) / (torch.sum(all_imgs_f0[0], dim=(1,2,3))[:, None, None, None] + 1e-6)
        all_nor_imgs_f0[1] = (all_imgs_f0[1] + 1e-6) / (torch.sum(all_imgs_f0[1], dim=(1,2,3))[:, None, None, None] + 1e-6)
        
        all_imgs0, all_imgs_f0, all_nor_imgs_f0 = scale_ops_2(all_imgs0, all_imgs_f0, all_nor_imgs_f0, args)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
            
        all_imgs_l = all_imgs_l.type(torch.LongTensor)

        inputs1 = all_imgs_f0[0].clone()
        for image_index in range(args.num_mixing-1):
            inputs1 =  torch.cat((inputs1, all_imgs_f0[image_index+1].clone()), 1)
            
        a = 0.0
        b = 1.0
        input_ratio_raw = ((b - a) * torch.rand(bs) + a).cuda(args.gpu)
        
        input_ratio = input_ratio_raw[:,None,None,None]
        input_ratio = input_ratio.repeat(1,1,args.image_size,args.image_size)
        inputs1 =  torch.cat((inputs1, input_ratio), 1)
        
        output = args.ATMModel(inputs1)
        mixing_matrix = output[:,:args.mixing_sectors*args.mixing_sectors]
        trans = output[:,args.mixing_sectors*args.mixing_sectors:]

        mixing_matrix = torch.reshape(mixing_matrix, (bs,1, args.mixing_sectors, args.mixing_sectors))
        mixing_matrix = (mixing_matrix + 1) / 2
        mixing_matrix = transforms.functional.resize(mixing_matrix, args.image_size).clone()
        
        all_imgs0,all_imgs_f0,all_nor_imgs_f0,xxx = output_augment_ratio_scale_2(trans, all_imgs0, all_imgs_f0, all_nor_imgs_f0, args)  
        
        mixed_imgs = all_imgs0[0]*mixing_matrix + all_imgs0[1]*(1-mixing_matrix)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f00')
            
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f01')
            
        if torch.isnan(mixing_matrix).any():
            print('mixing_matrix')

        ratio1 = torch.sum(all_nor_imgs_f0[0]*mixing_matrix, (1,2,3)).cuda(args.gpu)
        ratio2 = torch.sum(all_nor_imgs_f0[1]*(1-mixing_matrix), (1,2,3)).cuda(args.gpu)
        
        if torch.isnan(ratio1).any():
            print('ratio1')
        if torch.isnan(ratio2).any():
            print('ratio2')

        total_info = (ratio1 + ratio2).cuda(args.gpu)

        ratio1 = ratio1/total_info
        ratio2 = ratio2/total_info
        #print(ratio1+ratio2)
        #print(ratio1.get_device(), two_hot_labels.get_device())
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[0]] += ratio1
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[1]] += ratio2

        return mixed_imgs, two_hot_labels
    else:
        two_hot_labels[np.arange(imgs.shape[0]), labels[0:imgs.shape[0]]] = 1.0
        return imgs, two_hot_labels


def ATM_ratio_2(imgs, img_feature, labels, args):
    two_hot_labels = torch.zeros((imgs.shape[0], args.num_class)).cuda(args.gpu)

    if torch.rand(1) <= args.prob:

        bs = imgs.shape[0]
        
        rand_index = torch.randperm(imgs.size()[0])
        all_imgs0 = torch.zeros(args.num_mixing,bs,3,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_l = torch.zeros(args.num_mixing,bs).cuda(args.gpu)
        
        all_imgs0[0] = imgs.clone()
        all_imgs_f0[0] = img_feature.clone()
        all_imgs_l[0] = labels.clone()
        
        for image_index in range(args.num_mixing-1) :
            rand_index = torch.randperm(imgs.size()[0])
            all_imgs0[image_index+1] = imgs[rand_index].clone()
            all_imgs_f0[image_index+1] = img_feature[rand_index].clone()
            all_imgs_l[image_index+1] = labels[rand_index].clone()
            
        
            
        all_nor_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        if torch.isnan(all_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
        all_nor_imgs_f0[0] = (all_imgs_f0[0] + 1e-6) / (torch.sum(all_imgs_f0[0], dim=(1,2,3))[:, None, None, None] + 1e-6)
        all_nor_imgs_f0[1] = (all_imgs_f0[1] + 1e-6) / (torch.sum(all_imgs_f0[1], dim=(1,2,3))[:, None, None, None] + 1e-6)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
            
        all_imgs_l = all_imgs_l.type(torch.LongTensor)

        inputs1 = all_imgs_f0[0].clone()
        for image_index in range(args.num_mixing-1):
            inputs1 =  torch.cat((inputs1, all_imgs_f0[image_index+1].clone()), 1)
            
        a = 0.1
        b = 0.9
        input_ratio_raw = ((b - a) * torch.rand(bs) + a).cuda(args.gpu)
        
        input_ratio = input_ratio_raw[:,None,None,None]
        input_ratio = input_ratio.repeat(1,1,args.image_size,args.image_size)
        inputs1 =  torch.cat((inputs1, input_ratio), 1)
        
        output = args.ATMModel(inputs1)
        mixing_matrix = output[:,:args.mixing_sectors*args.mixing_sectors]
        trans = output[:,args.mixing_sectors*args.mixing_sectors:]

        mixing_matrix = torch.reshape(mixing_matrix, (bs,1, args.mixing_sectors, args.mixing_sectors))
        mixing_matrix = (mixing_matrix + 1) / 2
        mixing_matrix = transforms.functional.resize(mixing_matrix, args.image_size).clone()
        
        all_imgs0,all_imgs_f0,all_nor_imgs_f0,xxx = output_augment_ratio_2(trans, all_imgs0, all_imgs_f0, all_nor_imgs_f0, args)  
        
        mixed_imgs = all_imgs0[0]*mixing_matrix + all_imgs0[1]*(1-mixing_matrix)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f00')
            
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f01')
            
        if torch.isnan(mixing_matrix).any():
            print('mixing_matrix')

        ratio1 = torch.sum(all_nor_imgs_f0[0]*mixing_matrix, (1,2,3)).cuda(args.gpu)
        ratio2 = torch.sum(all_nor_imgs_f0[1]*(1-mixing_matrix), (1,2,3)).cuda(args.gpu)
        
        if torch.isnan(ratio1).any():
            print('ratio1')
        if torch.isnan(ratio2).any():
            print('ratio2')

        total_info = (ratio1 + ratio2).cuda(args.gpu)

        ratio1 = ratio1/total_info
        ratio2 = ratio2/total_info
        #print(ratio1+ratio2)
        #print(ratio1.get_device(), two_hot_labels.get_device())
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[0]] += ratio1
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[1]] += ratio2

        return mixed_imgs, two_hot_labels
    else:
        two_hot_labels[np.arange(imgs.shape[0]), labels[0:imgs.shape[0]]] = 1.0
        return imgs, two_hot_labels


def ATM_ratio_mask(imgs, img_feature, labels, args):
    two_hot_labels = torch.zeros((imgs.shape[0], args.num_class)).cuda(args.gpu)

    if torch.rand(1) <= args.prob:

        bs = imgs.shape[0]
        
        rand_index = torch.randperm(imgs.size()[0])
        all_imgs0 = torch.zeros(args.num_mixing,bs,3,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_l = torch.zeros(args.num_mixing,bs).cuda(args.gpu)
        
        all_imgs0[0] = imgs.clone()
        all_imgs_f0[0] = img_feature.clone()
        all_imgs_l[0] = labels.clone()
        
        for image_index in range(args.num_mixing-1) :
            rand_index = torch.randperm(imgs.size()[0])
            all_imgs0[image_index+1] = imgs[rand_index].clone()
            all_imgs_f0[image_index+1] = img_feature[rand_index].clone()
            all_imgs_l[image_index+1] = labels[rand_index].clone()
            
        
            
        all_nor_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        if torch.isnan(all_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
        all_nor_imgs_f0[0] = (all_imgs_f0[0] + 1e-6) / (torch.sum(all_imgs_f0[0], dim=(1,2,3))[:, None, None, None] + 1e-6)
        all_nor_imgs_f0[1] = (all_imgs_f0[1] + 1e-6) / (torch.sum(all_imgs_f0[1], dim=(1,2,3))[:, None, None, None] + 1e-6)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
            
        all_imgs_l = all_imgs_l.type(torch.LongTensor)

        inputs1 = all_imgs_f0[0].clone()
        for image_index in range(args.num_mixing-1):
            inputs1 =  torch.cat((inputs1, all_imgs_f0[image_index+1].clone()), 1)
            
        a = 0.0
        b = 1.0
        input_ratio_raw = ((b - a) * torch.rand(bs) + a).cuda(args.gpu)
        
        input_ratio = input_ratio_raw[:,None,None,None]
        input_ratio = input_ratio.repeat(1,1,args.image_size,args.image_size)
        inputs1 =  torch.cat((inputs1, input_ratio), 1)
        
        output = args.ATMModel(inputs1)
        mixing_matrix = output[:,:args.mixing_sectors*args.mixing_sectors]

        mixing_matrix = torch.reshape(mixing_matrix, (bs,1, args.mixing_sectors, args.mixing_sectors))
        mixing_matrix = (mixing_matrix + 1) / 2
        mixing_matrix = transforms.functional.resize(mixing_matrix, args.image_size).clone()
        
        mixed_imgs = all_imgs0[0]*mixing_matrix + all_imgs0[1]*(1-mixing_matrix)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f00')
            
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f01')
            
        if torch.isnan(mixing_matrix).any():
            print('mixing_matrix')

        ratio1 = torch.sum(all_nor_imgs_f0[0]*mixing_matrix, (1,2,3)).cuda(args.gpu)
        ratio2 = torch.sum(all_nor_imgs_f0[1]*(1-mixing_matrix), (1,2,3)).cuda(args.gpu)
        
        if torch.isnan(ratio1).any():
            print('ratio1')
        if torch.isnan(ratio2).any():
            print('ratio2')

        total_info = (ratio1 + ratio2).cuda(args.gpu)

        ratio1 = ratio1/total_info
        ratio2 = ratio2/total_info
        #print(ratio1+ratio2)
        #print(ratio1.get_device(), two_hot_labels.get_device())
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[0]] += ratio1
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[1]] += ratio2

        return mixed_imgs, two_hot_labels
    else:
        two_hot_labels[np.arange(imgs.shape[0]), labels[0:imgs.shape[0]]] = 1.0
        return imgs, two_hot_labels
    

def ATM_ratio_mask_no_nor(imgs, img_feature, labels, args):
    two_hot_labels = torch.zeros((imgs.shape[0], args.num_class)).cuda(args.gpu)

    if torch.rand(1) <= args.prob:

        bs = imgs.shape[0]
        
        rand_index = torch.randperm(imgs.size()[0])
        all_imgs0 = torch.zeros(args.num_mixing,bs,3,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_l = torch.zeros(args.num_mixing,bs).cuda(args.gpu)
        
        all_imgs0[0] = imgs.clone()
        all_imgs_f0[0] = img_feature.clone()
        all_imgs_l[0] = labels.clone()
        
        for image_index in range(args.num_mixing-1) :
            rand_index = torch.randperm(imgs.size()[0])
            all_imgs0[image_index+1] = imgs[rand_index].clone()
            all_imgs_f0[image_index+1] = img_feature[rand_index].clone()
            all_imgs_l[image_index+1] = labels[rand_index].clone()
            
        
            
        all_nor_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        if torch.isnan(all_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
        all_nor_imgs_f0[0] = (all_imgs_f0[0] + 1e-6) / (torch.sum(all_imgs_f0[0], dim=(1,2,3))[:, None, None, None] + 1e-6)
        all_nor_imgs_f0[1] = (all_imgs_f0[1] + 1e-6) / (torch.sum(all_imgs_f0[1], dim=(1,2,3))[:, None, None, None] + 1e-6)
        
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
            
        all_imgs_l = all_imgs_l.type(torch.LongTensor)

        inputs1 = all_imgs_f0[0].clone()
        for image_index in range(args.num_mixing-1):
            inputs1 =  torch.cat((inputs1, all_imgs_f0[image_index+1].clone()), 1)
            
        a = 0.0
        b = 1.0
        input_ratio_raw = ((b - a) * torch.rand(bs) + a).cuda(args.gpu)
        
        input_ratio = input_ratio_raw[:,None,None,None]
        input_ratio = input_ratio.repeat(1,1,args.image_size,args.image_size)
        inputs1 =  torch.cat((inputs1, input_ratio), 1)
        
        output = args.ATMModel(inputs1)
        mixing_matrix = output[:,:args.mixing_sectors*args.mixing_sectors]

        mixing_matrix = torch.reshape(mixing_matrix, (bs,1, args.mixing_sectors, args.mixing_sectors))
        mixing_matrix = (mixing_matrix + 1) / 2
        mixing_matrix = transforms.functional.resize(mixing_matrix, args.image_size).clone()
        
        mixed_imgs = all_imgs0[0]*mixing_matrix + all_imgs0[1]*(1-mixing_matrix)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f00')
            
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f01')
            
        if torch.isnan(mixing_matrix).any():
            print('mixing_matrix')

        ratio1 = torch.sum(all_nor_imgs_f0[0]*mixing_matrix, (1,2,3)).cuda(args.gpu)
        ratio2 = torch.sum(all_nor_imgs_f0[1]*(1-mixing_matrix), (1,2,3)).cuda(args.gpu)
        
        if torch.isnan(ratio1).any():
            print('ratio1')
        if torch.isnan(ratio2).any():
            print('ratio2')

        total_info = (ratio1 + ratio2).cuda(args.gpu)

        # ratio1 = ratio1/total_info
        # ratio2 = ratio2/total_info
        #print(ratio1+ratio2)
        #print(ratio1.get_device(), two_hot_labels.get_device())
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[0]] += ratio1
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[1]] += ratio2

        return mixed_imgs, two_hot_labels
    else:
        two_hot_labels[np.arange(imgs.shape[0]), labels[0:imgs.shape[0]]] = 1.0
        return imgs, two_hot_labels

def ATM_same(imgs, img_feature, labels, args):
    two_hot_labels = torch.zeros((imgs.shape[0], args.num_class)).cuda(args.gpu)

    if torch.rand(1) <= args.prob:

        bs = imgs.shape[0]
        
        rand_index = torch.randperm(imgs.size()[0])
        all_imgs0 = torch.zeros(args.num_mixing,bs,3,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_l = torch.zeros(args.num_mixing,bs).cuda(args.gpu)
        
        all_imgs0[0] = imgs.clone()
        all_imgs_f0[0] = img_feature.clone()
        all_imgs_l[0] = labels.clone()
        
        for image_index in range(args.num_mixing-1) :
            rand_index = torch.randperm(imgs.size()[0])
            all_imgs0[image_index+1] = imgs[rand_index].clone()
            all_imgs_f0[image_index+1] = img_feature[rand_index].clone()
            all_imgs_l[image_index+1] = labels[rand_index].clone()
            
        all_nor_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        if torch.isnan(all_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
        all_nor_imgs_f0[0] = (all_imgs_f0[0] + 1e-6) / (torch.sum(all_imgs_f0[0], dim=(1,2,3))[:, None, None, None] + 1e-6)
        all_nor_imgs_f0[1] = (all_imgs_f0[1] + 1e-6) / (torch.sum(all_imgs_f0[1], dim=(1,2,3))[:, None, None, None] + 1e-6)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
            
        all_imgs_l = all_imgs_l.type(torch.LongTensor)

        inputs1 = all_imgs_f0[0].clone()
        for image_index in range(args.num_mixing-1):
            inputs1 =  torch.cat((inputs1, all_imgs_f0[image_index+1].clone()), 2)
            
        output = args.ATMModel(inputs1)
        mixing_matrix = output[:,:args.mixing_sectors*args.mixing_sectors]
        trans = output[:,args.mixing_sectors*args.mixing_sectors:]

        mixing_matrix = torch.reshape(mixing_matrix, (bs,1, args.mixing_sectors, args.mixing_sectors))
        mixing_matrix = (mixing_matrix + 1) / 2
        mixing_matrix = transforms.functional.resize(mixing_matrix, args.image_size).clone()
        
        all_imgs0,all_imgs_f0,all_nor_imgs_f0,xxx = output_augment(trans, all_imgs0, all_imgs_f0, all_nor_imgs_f0, args)  
        
        mixed_imgs = all_imgs0[0]*mixing_matrix + all_imgs0[1]*(1-mixing_matrix)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f00')
            
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f01')
            
        if torch.isnan(mixing_matrix).any():
            print('mixing_matrix')

        ratio1 = torch.sum(all_nor_imgs_f0[0]*mixing_matrix, (1,2,3)).cuda(args.gpu)
        ratio2 = torch.sum(all_nor_imgs_f0[1]*(1-mixing_matrix), (1,2,3)).cuda(args.gpu)
        
        if torch.isnan(ratio1).any():
            print('ratio1')
        if torch.isnan(ratio2).any():
            print('ratio2')

        total_info = (ratio1 + ratio2).cuda(args.gpu)

        ratio1 = ratio1/total_info
        ratio2 = ratio2/total_info
        #print(ratio1+ratio2)
        #print(ratio1.get_device(), two_hot_labels.get_device())
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[0]] += ratio1
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[1]] += ratio2

        return mixed_imgs, two_hot_labels
    else:
        two_hot_labels[np.arange(imgs.shape[0]), labels[0:imgs.shape[0]]] = 1.0
        return imgs, two_hot_labels
    
def ATM_scale(imgs, img_feature, labels, args):
    two_hot_labels = torch.zeros((imgs.shape[0], args.num_class)).cuda(args.gpu)

    if torch.rand(1) <= args.prob:

        bs = imgs.shape[0]
        

        
        rand_index = torch.randperm(imgs.size()[0])
        all_imgs0 = torch.zeros(args.num_mixing,bs,3,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_l = torch.zeros(args.num_mixing,bs).cuda(args.gpu)
        
        all_imgs0[0] = imgs.clone()
        all_imgs_f0[0] = img_feature.clone()
        all_imgs_l[0] = labels.clone()
        
        for image_index in range(args.num_mixing-1) :
            rand_index = torch.randperm(imgs.size()[0])
            all_imgs0[image_index+1] = imgs[rand_index].clone()
            all_imgs_f0[image_index+1] = img_feature[rand_index].clone()
            all_imgs_l[image_index+1] = labels[rand_index].clone()
            
        all_imgs0, all_imgs_f0 = scale_ops(all_imgs0, all_imgs_f0, args)
            
        all_nor_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        if torch.isnan(all_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
        all_nor_imgs_f0[0] = (all_imgs_f0[0] + 1e-6) / (torch.sum(all_imgs_f0[0], dim=(1,2,3))[:, None, None, None] + 1e-6)
        all_nor_imgs_f0[1] = (all_imgs_f0[1] + 1e-6) / (torch.sum(all_imgs_f0[1], dim=(1,2,3))[:, None, None, None] + 1e-6)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
            
        all_imgs_l = all_imgs_l.type(torch.LongTensor)

        inputs1 = all_imgs_f0[0].clone()
        for image_index in range(args.num_mixing-1):
            inputs1 =  torch.cat((inputs1, all_imgs_f0[image_index+1].clone()), 2)
            
        output = args.ATMModel_M(inputs1)
        mixing_matrix = torch.reshape(output[:,:args.mixing_sectors*args.mixing_sectors], (bs, 1,args.image_size,args.image_size))
        
        trans = args.ATMModel_T(torch.cat((all_nor_imgs_f0[0],all_nor_imgs_f0[1],mixing_matrix),2))

        mixing_matrix = torch.reshape(mixing_matrix, (bs,1, args.mixing_sectors, args.mixing_sectors))
        mixing_matrix = (mixing_matrix + 1) / 2
        mixing_matrix = transforms.functional.resize(mixing_matrix, args.image_size).clone()
        
        all_imgs0,all_imgs_f0,all_nor_imgs_f0,xxx = output_augment_scale(trans, all_imgs0, all_imgs_f0, all_nor_imgs_f0, args)  
        
        mixed_imgs = all_imgs0[0]*mixing_matrix + all_imgs0[1]*(1-mixing_matrix)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f00')
            
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f01')
            
        if torch.isnan(mixing_matrix).any():
            print('mixing_matrix')

        ratio1 = torch.sum(all_nor_imgs_f0[0]*mixing_matrix, (1,2,3)).cuda(args.gpu)
        ratio2 = torch.sum(all_nor_imgs_f0[1]*(1-mixing_matrix), (1,2,3)).cuda(args.gpu)
        
        if torch.isnan(ratio1).any():
            print('ratio1')
        if torch.isnan(ratio2).any():
            print('ratio2')

        total_info = (ratio1 + ratio2).cuda(args.gpu)

        ratio1 = ratio1/total_info
        ratio2 = ratio2/total_info
        #print(ratio1+ratio2)
        #print(ratio1.get_device(), two_hot_labels.get_device())
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[0]] += ratio1
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[1]] += ratio2

        return mixed_imgs, two_hot_labels
    else:
        two_hot_labels[np.arange(imgs.shape[0]), labels[0:imgs.shape[0]]] = 1.0
        return imgs, two_hot_labels
    

def ATM_alt(imgs, img_feature, labels, args):
    two_hot_labels = torch.zeros((imgs.shape[0], args.num_class)).cuda(args.gpu)

    if torch.rand(1) <= args.prob:

        bs = imgs.shape[0]
        

        
        rand_index = torch.randperm(imgs.size()[0])
        all_imgs0 = torch.zeros(args.num_mixing,bs,3,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_l = torch.zeros(args.num_mixing,bs).cuda(args.gpu)
        
        all_imgs0[0] = imgs.clone()
        all_imgs_f0[0] = img_feature.clone()
        all_imgs_l[0] = labels.clone()
        
        for image_index in range(args.num_mixing-1) :
            rand_index = torch.randperm(imgs.size()[0])
            all_imgs0[image_index+1] = imgs[rand_index].clone()
            all_imgs_f0[image_index+1] = img_feature[rand_index].clone()
            all_imgs_l[image_index+1] = labels[rand_index].clone()
            
        all_nor_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        if torch.isnan(all_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
        all_nor_imgs_f0[0] = (all_imgs_f0[0] + 1e-6) / (torch.sum(all_imgs_f0[0], dim=(1,2,3))[:, None, None, None] + 1e-6)
        all_nor_imgs_f0[1] = (all_imgs_f0[1] + 1e-6) / (torch.sum(all_imgs_f0[1], dim=(1,2,3))[:, None, None, None] + 1e-6)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
            
        all_imgs_l = all_imgs_l.type(torch.LongTensor)

        inputs1 = all_imgs_f0[0].clone()
        for image_index in range(args.num_mixing-1):
            inputs1 =  torch.cat((inputs1, all_imgs_f0[image_index+1].clone()), 2)
            
        output = args.ATMModel_M(inputs1)
        mixing_matrix = torch.reshape(output[:,:args.mixing_sectors*args.mixing_sectors], (bs, 1,args.image_size,args.image_size))
        
        trans = args.ATMModel_T(torch.cat((all_nor_imgs_f0[0],all_nor_imgs_f0[1],mixing_matrix),2))

        mixing_matrix = torch.reshape(mixing_matrix, (bs,1, args.mixing_sectors, args.mixing_sectors))
        mixing_matrix = (mixing_matrix + 1) / 2
        mixing_matrix = transforms.functional.resize(mixing_matrix, args.image_size).clone()
        
        all_imgs0,all_imgs_f0,all_nor_imgs_f0,xxx = output_augment_scale(trans, all_imgs0, all_imgs_f0, all_nor_imgs_f0, args)  
        
        mixed_imgs = all_imgs0[0]*mixing_matrix + all_imgs0[1]*(1-mixing_matrix)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f00')
            
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f01')
            
        if torch.isnan(mixing_matrix).any():
            print('mixing_matrix')

        ratio1 = torch.sum(all_nor_imgs_f0[0]*mixing_matrix, (1,2,3)).cuda(args.gpu)
        ratio2 = torch.sum(all_nor_imgs_f0[1]*(1-mixing_matrix), (1,2,3)).cuda(args.gpu)
        
        if torch.isnan(ratio1).any():
            print('ratio1')
        if torch.isnan(ratio2).any():
            print('ratio2')

        total_info = (ratio1 + ratio2).cuda(args.gpu)

        ratio1 = ratio1/total_info
        ratio2 = ratio2/total_info
        #print(ratio1+ratio2)
        #print(ratio1.get_device(), two_hot_labels.get_device())
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[0]] += ratio1
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[1]] += ratio2

        return mixed_imgs, two_hot_labels
    else:
        two_hot_labels[np.arange(imgs.shape[0]), labels[0:imgs.shape[0]]] = 1.0
        return imgs, two_hot_labels
    
def ATM_mask(imgs, img_feature, labels, args):
    two_hot_labels = torch.zeros((imgs.shape[0], args.num_class)).cuda(args.gpu)

    if torch.rand(1) <= args.prob:

        bs = imgs.shape[0]
        
        rand_index = torch.randperm(imgs.size()[0])
        all_imgs0 = torch.zeros(args.num_mixing,bs,3,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        all_imgs_l = torch.zeros(args.num_mixing,bs).cuda(args.gpu)
        
        all_imgs0[0] = imgs.clone()
        all_imgs_f0[0] = img_feature.clone()
        all_imgs_l[0] = labels.clone()
        
        for image_index in range(args.num_mixing-1) :
            rand_index = torch.randperm(imgs.size()[0])
            all_imgs0[image_index+1] = imgs[rand_index].clone()
            all_imgs_f0[image_index+1] = img_feature[rand_index].clone()
            all_imgs_l[image_index+1] = labels[rand_index].clone()
            
        all_nor_imgs_f0 = torch.zeros(args.num_mixing,bs,1,args.image_size,args.image_size).cuda(args.gpu)
        if torch.isnan(all_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
        all_nor_imgs_f0[0] = (all_imgs_f0[0] + 1e-6) / (torch.sum(all_imgs_f0[0], dim=(1,2,3))[:, None, None, None] + 1e-6)
        all_nor_imgs_f0[1] = (all_imgs_f0[1] + 1e-6) / (torch.sum(all_imgs_f0[1], dim=(1,2,3))[:, None, None, None] + 1e-6)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f000')
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f010')
            
        all_imgs_l = all_imgs_l.type(torch.LongTensor)

        inputs1 = all_imgs_f0[0].clone()
        for image_index in range(args.num_mixing-1):
            inputs1 =  torch.cat((inputs1, all_imgs_f0[image_index+1].clone()), 2)
            
        mixing_matrix = args.ATMModel(inputs1)
        mixing_matrix = torch.reshape(mixing_matrix, (bs,1, args.mixing_sectors, args.mixing_sectors))
        mixing_matrix = (mixing_matrix + 1) / 2
        mixing_matrix = transforms.functional.resize(mixing_matrix, args.image_size).clone()

        inputs2 = all_imgs_f0[0]
        mm = mixing_matrix.clone().detach()
        for image_index in range(args.num_mixing-1):
            inputs2 =  torch.cat((inputs2, all_imgs_f0[image_index+1]), 2)
        inputs2 = torch.cat((inputs2, mm), 2)
        
        mixed_imgs = all_imgs0[0]*mixing_matrix + all_imgs0[1]*(1-mixing_matrix)
        
        if torch.isnan(all_nor_imgs_f0[0]).any():
            print('all_nor_imgs_f00')
            
        if torch.isnan(all_nor_imgs_f0[1]).any():
            print('all_nor_imgs_f01')
            
        if torch.isnan(mixing_matrix).any():
            print('mixing_matrix')

        ratio1 = torch.sum(all_nor_imgs_f0[0]*mixing_matrix, (1,2,3)).cuda(args.gpu)
        ratio2 = torch.sum(all_nor_imgs_f0[1]*(1-mixing_matrix), (1,2,3)).cuda(args.gpu)
        
        if torch.isnan(ratio1).any():
            print('ratio1')
        if torch.isnan(ratio2).any():
            print('ratio2')

        total_info = (ratio1 + ratio2).cuda(args.gpu)

        ratio1 = ratio1/total_info
        ratio2 = ratio2/total_info
        #print(ratio1+ratio2)
        #print(ratio1.get_device(), two_hot_labels.get_device())
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[0]] += ratio1
        two_hot_labels[np.arange(imgs.shape[0]), all_imgs_l[1]] += ratio2

        return mixed_imgs, two_hot_labels
    else:
        two_hot_labels[np.arange(imgs.shape[0]), labels[0:imgs.shape[0]]] = 1.0
        return imgs, two_hot_labels


def aug_method(name):
    if name == 'baseline':
        return baseline
    elif name == 'MixUp':
        return MixUp
    elif name == 'CutMix':
        return CutMix
    elif name == 'SaliencyMix':
        return SaliencyMix
    elif name == 'SmoothMix':
        return SmoothMix
    elif name == 'ResizeMix':
        return ResizeMix
    elif name == 'ATM':
        return ATM
    elif name == 'ATM_alt':
        return ATM_alt
    elif name == 'ATM_mask':
        return ATM_mask
    elif name ==  'ATM_scale':
        return ATM_scale
    elif name ==  'ATM_dim':
        return ATM_dim
    elif name == 'ATM_ratio':
        return ATM_ratio
    elif name == 'ATM_ratio_scale':
        return ATM_ratio_scale
    elif name == 'ATM_ratio_scale_2':
        return ATM_ratio_scale_2
    elif name == 'ATM_ratio_2':
        return ATM_ratio_2
    elif name == 'ATM_ratio_mask':
        return ATM_ratio_mask
    elif name == 'ATM_ratio_mask_no_nor':
        return ATM_ratio_mask_no_nor



