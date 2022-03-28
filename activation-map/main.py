import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from ipywidgets import IntProgress
from torchvision import models
from easydict import EasyDict
import matplotlib.pyplot as plt
args = EasyDict()
from tqdm import tqdm
from PIL import Image
import os
from torchvision import transforms
from dataset import CIFAR10,CIFAR100
import argparse
from collections import OrderedDict

import sys
sys.path.append('../')
from models import *

dataset_options = ['cifar10', 'cifar100', 'imagenet']
model_options = ['resnet18', 'densenet161', 'VGG19']
method_options = [ 'gradcam', 'scorecam', 'gradcam++', 'ablationcam', 'xgradcam' \
                    'eigencam', 'eigengradcam', 'layercam', 'fullgrad']

parser = argparse.ArgumentParser(description='MultiMix')
parser.add_argument('--dataset', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--model', default='resnet18',
                    choices=model_options)
parser.add_argument('--method', default='gradcam',
                    choices=method_options)
parser.add_argument('--model_dir', 
    default='../checkpoints/cifar10-resnet18-baseline.pth', type=str)
parser.add_argument('--save_dir', 
    default='../data/cifar10-resnet18-baseline-gradcam', type=str)
args = parser.parse_args()

args.methods = \
    {"gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM}

args.use_cuda = True
args.aug_smooth=True
args.eigen_smooth=True
#os.environ["CUDA_LAUNCH_BLOCKING"]="1"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

args.gpu = 0

if args.dataset == 'cifar10':
    num_classes = 10
elif args.dataset =='cifar100':
    num_classes = 100
elif args.dataset =='imagenet':
    num_classes = 1000



if args.model == 'resnet18':
    model = ResNet18(num_classes=num_classes)
elif args.model == 'resnet50':
    model = ResNet50(num_classes=num_classes)
elif args.model == 'densenet161':
    model = DenseNet161(num_classes=num_classes)
elif args.model == 'VGG19':
    model = VGG('VGG19', num_classes=num_classes)

# state_dict = torch.load(args.model_dir)
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:]
#     new_state_dict[name] = v
# model.load_state_dict(state_dict)
model = torch.load(args.model_dir)
# print(model)
model.eval().cuda()

# FasterRCNN: model.backbone
# Resnet18 and 50: model.layer4[-1]
# VGG and densenet161: model.features[-1]
# mnasnet1_0: model.layers[-1]
# ViT: model.blocks[-1].norm1
# SwinT: model.layers[-1].blocks[-1].norm1

if args.model == 'resnet18':
    target_layers = model.layer4[-1]   
elif args.model == 'resnet50':
    target_layers = model.layer4[-1]
elif args.model == 'densenet161':
    target_layers = model.dense4[-1]
elif args.model == 'VGG19':
    target_layers = model.features[49]
    
# target_layers = model.layer4[-1]
# target_layers = model.dense4[-1]


cam_algorithm = args.methods[args.method]
cam = cam_algorithm(model=model, target_layers=target_layers, args=args, use_cuda=True)
#cam = cam_algorithm(model=model, target_layer=target_layers, use_cuda=True)

if args.dataset == 'cifar100':
    image_size = 32
    mean = [0.5070, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2761]
    num_class = 100
    
if args.dataset == 'cifar10':
    image_size = 32
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.247, 0.243, 0.261]
    num_class = 10

batch_size = 64
workers = 16
args.batch_size = batch_size
args.workers = workers

transform = transforms.ToTensor()
transform2 = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

if args.dataset == 'cifar10':
    args.num_class = num_classes = 10
    train_dataset = CIFAR10(root='../data',
                                     train=True,
                                     transform=None,
                                     download=True)

    val_dataset = CIFAR10(root='../data',
                                    train=False,
                                    transform=None,
                                    download=True)
elif args.dataset == 'cifar100':
    args.num_class = num_classes = 100
    train_dataset = CIFAR100(root='../data',
                                      train=True,
                                      transform=None,
                                      download=True)

    val_dataset = CIFAR100(root='../data',
                                     train=False,
                                     transform=None,
                                     download=True)


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

train_path = args.save_dir + '/train'

if not os.path.exists(train_path):
    os.makedirs(train_path)

val_path = args.save_dir + '/val'
    
if not os.path.exists(val_path):
    os.makedirs(val_path)
    
for k in range(args.num_class):
    if not os.path.exists(train_path+'/'+str(k)):
        os.makedirs(train_path+'/'+str(k))
        
for k in range(args.num_class):
    if not os.path.exists(val_path+'/'+str(k)):
        os.makedirs(val_path+'/'+str(k))
        
def process_data_single(dataset, args):
    
    if args.train_dataset:
        pp = train_path
    else:
        pp = val_path

    indexes = torch.zeros(num_class)
    
    t = tqdm(dataset)

    for i, data in enumerate(dataset, 0):
        input_raw, label = data[0], data[1]
        
        index = int(indexes[label])
        #print(type(input_raw), input_raw.size)
        im = input_raw#Image.fromarray(input_raw)
        np.save(pp+'/'+str(label)+'/'+str(index) + '.npy', im)
        input = Image.fromarray(input_raw)
        input = transform2(transform(input_raw))

        grayscale_cam = cam(input_tensor=torch.unsqueeze(input,0),
                                target_category=label)
        
        #print(type(grayscale_cam),grayscale_cam.shape)
        grayscale_cam = (grayscale_cam*255).astype(np.uint8)
        grayscale_cam = np.squeeze(grayscale_cam)
        #print(type(grayscale_cam),grayscale_cam.shape)
        im = Image.fromarray(grayscale_cam)
        aa = im.save(pp+'/'+str(label)+'/'+str(index)+'f.JPEG' )
        
            
        if args.train_dataset:
            with open(args.save_dir + '/train.txt', 'a') as f:
                f.writelines(pp[3:]+'/'+str(label)+'/'+str(index)+ '.npy\n')
            with open(args.save_dir + '/trainf.txt', 'a') as f:
                f.writelines(pp[3:]+'/'+str(label)+'/'+str(index)+ 'f.JPEG\n')
        else:
            with open(args.save_dir + '/val.txt', 'a') as f:
                f.writelines(pp[3:]+'/'+str(label)+'/'+str(index)+ '.npy\n')
            with open(args.save_dir + '/valf.txt', 'a') as f:
                f.writelines(pp[3:]+'/'+str(label)+'/'+str(index)+ 'f.JPEG\n')
                
        indexes[label] = indexes[label] + 1

        t.update()
    t.close()


args.train_dataset = True
process_data_single(train_dataset, args)

args.train_dataset = False
process_data_single(val_dataset, args)