'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import os
import argparse

import sys
sys.path.append('../')
sys.path.append('../../')

from models import *
from tqdm import tqdm
from dataset import data_with_feature

from augmentation import aug_method
from randmix import RandMix
augmentation_options = ['baseline', 'MixUp', 'CutMix', 'ResizeMix', \
                        'SaliencyMix']
model_options = ['resnet18', 'wideresnet2810', 'VGG19','densenet161']
dataset_options = ['cifar10', 'cifar100']

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
parser.add_argument('--model', default='resnet18',choices=model_options)
parser.add_argument('--dataset', default='cifar10', type=str, choices=dataset_options)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
parser.add_argument('--num_workers', default=16, type=int, help='num_workers')
parser.add_argument('--epochs', default=200, type=int, help='epochs')
parser.add_argument('--augmentation', default='baseline',choices=augmentation_options)
parser.add_argument('--prob', default=0.5, type=float, help='augmentation prob')
parser.add_argument('--gpu', default=0, type=int, help='use which gpu')
parser.add_argument('--short_verbose', action='store_true')
args = parser.parse_args()

args.aug = aug_method(args.augmentation)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data

if args.dataset.startswith('cifar10'):
    args.num_class = 10
    transform_normal = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
if args.dataset.startswith('cifar100'):
    args.num_class = 100
    transform_normal = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

transform_spatial = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])

transform_train = transforms.Compose([
    transform_spatial,
    transform_normal,
])
if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(
        root='data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, 
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform_normal)
    testloader = torch.utils.data.DataLoader(testset, 
        batch_size=100, shuffle=False, num_workers=args.num_workers, pin_memory=True)
elif args.dataset == 'cifar100': 
    trainset = torchvision.datasets.CIFAR100(
        root='data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, 
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(
        root='data', train=False, download=True, transform=transform_normal)
    testloader = torch.utils.data.DataLoader(testset, 
        batch_size=100, shuffle=False, num_workers=args.num_workers, pin_memory=True)

# Model
# net = VGG('VGG19')
if args.model == 'resnet18':
    args.net = ResNet18(num_classes = args.num_class)
elif args.model == 'wideresnet2810':
    args.net = WideResNet(depth=28, num_classes=args.num_class, widen_factor=10,
                         dropRate=0.3)
elif args.model == 'VGG19':
    args.net = VGG('VGG19',num_classes = args.num_class)
elif args.model == 'densenet161':
    args.net = DenseNet161(num_classes = args.num_class)

args.net = args.net.to(device)

criterion = nn.CrossEntropyLoss()
args.optimizer = optim.SGD(args.net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(args.optimizer, T_max=args.epochs)


# Training
def train(epoch):
    
    if not args.short_verbose:
        t = tqdm(trainloader)
        t.set_description('Train ' + str(epoch))
    
    args.net.train()
    train_loss = 0.0
    for batch_idx, data in enumerate(trainloader):
        
        inputs, targets = data
            
        inputs, targets = inputs.to(device), targets.to(device)
            
        inputs, targets = args.aug(inputs, targets, args)
            
        args.optimizer.zero_grad()
        outputs = args.net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        args.optimizer.step()

        train_loss += loss.item()
        
        if not args.short_verbose:
            t.set_postfix(
                        l='%.3f' % (train_loss/(batch_idx+1))
                        )
            t.update()
    if not args.short_verbose:
        t.close()
        
    return train_loss/(batch_idx+1)


def test(epoch):
    global best_acc
    
    if not args.short_verbose:
        t = tqdm(testloader)
        t.set_description('Test ' + str(epoch))
    
    args.net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = args.net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if not args.short_verbose:
                t.set_postfix(
                        loss='%.3f' % (test_loss/(batch_idx+1)),
                        acc='%.3f' % (100.*correct/total)
                        )
                t.update()
           
        if not args.short_verbose: 
            t.close()

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        # print('Saving..')
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(args.net, './checkpoints/' + \
            args.dataset + '-' + args.model + '-' + \
            args.augmentation + '.pth')
        best_acc = acc
    
    return (test_loss/(batch_idx+1)), acc



if args.short_verbose: 
    t = tqdm(range(args.epochs))
    t.set_description('Train')
for epoch in range(args.epochs):
    
    loss = train(epoch)
    test_loss, acc = test(epoch)
    scheduler.step()
    
    if args.short_verbose:
        t.set_postfix(
                l1='%.3f' % loss,
                l2='%.3f' % test_loss,
                acc='%.3f' % acc
                )
        t.update()
           
if args.short_verbose: 
    t.close()
    
print('-'*20)
print('Best Accuracy:' + str(best_acc))
print('-'*20)

