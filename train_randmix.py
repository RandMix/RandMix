'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
import sys
sys.path.append('../')
sys.path.append('../../')

from models import *
from tqdm import tqdm
from dataset import data_with_feature

from randmix import RandMix

model_options = ['resnet18', 'wideresnet2810']
parser = argparse.ArgumentParser(description='PyTorch RandMix Training')
parser.add_argument('--model', default='resnet18',choices=model_options)
parser.add_argument('--dataset', default='cifar-100-resnet18-noMix', type=str)

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
parser.add_argument('--num_workers', default=16, type=int, help='num_workers')
parser.add_argument('--epochs', default=200, type=int, help='epochs')
parser.add_argument('--gpu', default=0, type=int, help='use which gpu')

parser.add_argument('--short_verbose', action='store_true')

parser.add_argument('--model_dir', type=str)

parser.add_argument('--rand_aug', action='store_true')
parser.add_argument('--auto_aug', action='store_true')

parser.add_argument('--n', default=2, type=int)
parser.add_argument('--p', default=0.5, type=float)
parser.add_argument('--alpha', default=1.0, type=float, help='aug alpha')

args = parser.parse_args()

rm = RandMix(args.n,args.p)

args.beta = args.alpha
args.image_size = 32
args.device = args.gpu = device = torch.device("cuda:0")
    
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
elif args.dataset.startswith('cifar-10'):
    args.num_class = 10
    transform_normal = transforms.Compose([
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
if args.dataset.startswith('cifar100'):
    args.num_class = 100
    transform_normal = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
elif args.dataset.startswith('cifar-100'):
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

FEATURE_DATA = False
    
if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, 
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_normal)
    testloader = torch.utils.data.DataLoader(testset, 
        batch_size=100, shuffle=False, num_workers=args.num_workers, pin_memory=True)
elif args.dataset == 'cifar100': 
    trainset = torchvision.datasets.CIFAR100(
        root='../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, 
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(
        root='../data', train=False, download=True, transform=transform_normal)
    testloader = torch.utils.data.DataLoader(testset, 
        batch_size=100, shuffle=False, num_workers=args.num_workers, pin_memory=True)
elif args.dataset.startswith('cifar10-') or args.dataset.startswith('cifar100-'): 
    FEATURE_DATA = True
    trainset = data_with_feature(root_dir='data/'+args.dataset,
        is_train=True, transform=transform_normal, ex_transform=transform_spatial, rand_aug=args.rand_aug)
    trainloader = torch.utils.data.DataLoader(trainset, 
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    testset = data_with_feature(root_dir='data/'+args.dataset,
        is_train=False, transform=transform_normal, ex_transform=None)
    testloader = torch.utils.data.DataLoader(dataset=testset, 
        batch_size=100, shuffle=False, num_workers=args.num_workers, pin_memory=True)

# Model
# net = VGG('VGG19')
if args.model == 'resnet18':
    args.net = ResNet18(num_classes = args.num_class)
elif args.model == 'wideresnet2810':
    args.net = WideResNet(28, args.num_class, 10, 0.3)
elif args.model == 'VGG19':
    args.net = VGG('VGG19',num_classes = args.num_class)
elif args.model == 'densenet161':
    args.net = DenseNet161(num_classes = args.num_class)
# net = PreActResNet18()
args.net = args.net.to(device)

criterion = nn.CrossEntropyLoss()
args.optimizer = optim.SGD(args.net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(args.optimizer, T_max=args.epochs)

# Training
def train(epoch):
    
    if not args.short_verbose:
        t = tqdm(trainloader)
        t.set_description('Train ' + str(epoch))
    
    args.net.train()
    train_loss = 0.0
    for batch_idx, data in enumerate(trainloader):
        I1, F1, L1 = data[0], data[1], data[2]
        I1 = I1.to(args.device)
        F1 = F1.to(args.device)
        L1 = L1.to(args.device)

        L1 = F.one_hot(L1, num_classes=args.num_class).float()
        
        I1,L1 = rm(I1,F1,L1,args)

        input = I1
        labels = L1
        
        # print(labels.shape)
        
        args.optimizer.zero_grad()
        outputs = args.net(input)
        
        loss = criterion(outputs, labels)
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
            inputs, _, targets = data
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
    
with open('records/record_' + args.dataset + '_' + args.model +  '.txt', 'a') as f:
    f.writelines('-'*30 + '\n')
    f.writelines('dataset       :' + args.dataset + '\n')
    f.writelines('model         :' + args.model + '\n')
    f.writelines('epochs        :' + str(args.epochs) + '\n')
    f.writelines('batch_size    :' + str(args.batch_size) + '\n')
    f.writelines('best_accuracy :' + str(best_acc) + '\n') 
    f.writelines('-'*30 + '\n')
    


