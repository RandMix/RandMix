import torch
import random
import torchvision
import numpy as np
from randmix_utils import *
import matplotlib.pyplot as plt

def Identity(imgs, original_imgs, acts, original_acts, original_labs, prob, args):
    acts_s = acts
    acts_p = torch.zeros_like(acts)
    return imgs, acts, acts_s, acts_p

def ResizeMix(imgs, original_imgs, acts, original_acts, original_labs, prob, args):
    
    rand_index = torch.randperm(imgs.size()[0]).cuda()
    if torch.rand(1) <= prob:
        
        image_size = imgs.shape[3]
        

        new_imgs = original_imgs[rand_index,:,:,:]
        new_acts = original_acts[rand_index,:,:,:]
        # new_labs = original_labs[rand_index,:]

        # a = 0.1
        # b = 0.8
        # scale_rate = ((b - a) * torch.rand(1) + a).cuda()
        
        scale_rate = torch.tensor(np.random.beta(args.alpha, args.beta)).cuda()
        # print(scale_rate)
        
        new_size = int(torch.ceil(scale_rate * image_size))
        new_imgs = torchvision.transforms.functional.resize(new_imgs, new_size)
        new_acts = torchvision.transforms.functional.resize(new_acts, new_size)

        upper = image_size-new_size
        if upper == 0:
            start_point_x = 0
            start_point_y = 0
        else:
            start_point_x = torch.randint(0, upper, (1,)).cuda()
            start_point_y = torch.randint(0, upper, (1,)).cuda()

        imgs[:,:,start_point_x:start_point_x + new_size, start_point_y:start_point_y + new_size] = new_imgs[:,:,:,:]

        acts[:,:,start_point_x:start_point_x + new_size, start_point_y:start_point_y + new_size] = new_acts[:,:,:,:]

        mask = torch.ones_like(acts)
        mask[:,:,start_point_x:start_point_x + new_size, start_point_y:start_point_y + new_size] = 0.0
        
        acts_s = acts.clone()
        acts_s[:,:,start_point_x:start_point_x + new_size, start_point_y:start_point_y + new_size] = 0.0
        

        acts_p = acts.clone() - acts_s.clone()
        
    else:
        mask = torch.ones_like(acts)
        acts_s = acts
        acts_p = torch.zeros_like(acts)

    return imgs, mask, acts, acts_s, acts_p, rand_index
        
def SaliencyMix(imgs, original_imgs, acts, original_acts, original_labs, prob, args):
    
    rand_index = torch.randperm(imgs.shape[0]).cuda()
    if torch.rand(1) <= prob:
        
        lam = np.random.beta(args.alpha, args.beta)
        
        
        bbx1, bby1, bbx2, bby2 = saliency_bbox(original_imgs[rand_index[0]], lam)
        imgs[:, :, bbx1:bbx2, bby1:bby2] = original_imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
        acts[:, :, bbx1:bbx2, bby1:bby2] = original_acts[rand_index, :, bbx1:bbx2, bby1:bby2]
        
        # new_labs = original_labs[rand_index,:]
        # labs = labs * lam + new_labs * (1-lam)
        
        mask = torch.ones_like(acts)
        mask[:, :, bbx1:bbx2, bby1:bby2] = 0.0
        acts_s = acts.clone()
        acts_s[:, :, bbx1:bbx2, bby1:bby2] = 0.0
        acts_p = acts.clone() - acts_s.clone()
    else:
        mask = torch.ones_like(acts)
        acts_s = acts
        acts_p = torch.zeros_like(acts)
        
    return imgs, mask, acts, acts_s, acts_p, rand_index

def MixUp(imgs, original_imgs, acts, original_acts, original_labs, prob, args):
    
    rand_index = torch.randperm(imgs.size()[0]).cuda()
    if torch.rand(1) <= prob:
            
        new_imgs = original_imgs[rand_index,:,:,:]

        lam = np.random.beta(args.alpha, args.beta)
        imgs = lam*imgs + (1-lam)*new_imgs
        
        mask = torch.ones_like(acts)
        mask = mask * lam
        acts_s = acts.clone()
        acts_s = acts_s * lam
        acts_p = acts.clone() - acts_s.clone()
    else:
        mask = torch.ones_like(acts)
        acts_s = acts
        acts_p = torch.zeros_like(acts)
        
    return imgs, mask, acts, acts_s, acts_p, rand_index

def FMix(imgs, original_imgs, acts, original_acts, original_labs, prob, args):
    
    rand_index = torch.randperm(imgs.size()[0]).cuda()
    if torch.rand(1) <= prob:
            
        lam, mask = sample_mask(alpha=args.alpha, b=args.beta, decay_power=3, shape=(32,32), max_soft=0.0, reformulate=False)
        lam, mask = torch.tensor(lam,dtype=torch.float).cuda(), torch.tensor(mask,dtype=torch.float).cuda()
        # print(rand_index[:5])
        new_imgs = original_imgs[rand_index,:,:,:]
        new_acts = original_acts[rand_index,:,:,:]
        # new_labs = original_labs[rand_index,:]
        

        imgs = mask*imgs + (1-mask)*new_imgs
        acts = mask*acts + (1-mask)*new_acts
        # labs = lam*labs + (1-lam)*new_labs
        
        acts_s = acts.clone()
        acts_s = acts_s*mask
        acts_p = acts.clone() - acts_s.clone()
    else:
        mask = torch.ones_like(acts)
        acts_s = acts
        acts_p = torch.zeros_like(acts)
    
    return imgs, mask, acts, acts_s, acts_p, rand_index
    
def augment_list():
    
    l = [
        (ResizeMix, 1.0),
        (SaliencyMix, 1.0),
        (MixUp, 1.0),
        (FMix, 1.0),
    ]

    return l



class RandMix:
    def __init__(self, n, cp):
        self.n = n
        self.cp = cp    # coefficient of probability
        self.augment_list = augment_list()

    def __call__(self, imgs, acts, labs, args):
        original_imgs = imgs
        original_labs = labs
        original_acts = acts
        ops = random.choices(self.augment_list, k=self.n)
        
        
        if self.n == 1:
            
            for op, p in ops:
                prob = self.cp * p
                imgs, _, acts, acts_s, acts_p, rand_index = op(imgs, original_imgs, acts, original_acts, original_labs, prob, args)
                acts_s = torch.sum(acts_s, (1,2,3))
                acts_p = torch.sum(acts_p, (1,2,3))
                total_activation = acts_s + acts_p
                ratio = (acts_s / total_activation)[:,None]
                
                labs = labs * ratio + labs[rand_index] * (1-ratio)
            
        elif self.n == 2:
            
            for i, (op, p) in enumerate(ops):
                
                # print(i,op)
                
                if i == 0:
                    
                    prob = self.cp * p
                    # val = (float(self.m) / 30) * float(maxval - minval) + minval
                    imgs, _, acts, acts_s1, acts_p1, rand_index1 = op(imgs, original_imgs, acts, original_acts, original_labs, prob, args)
                    # print(torch.sum(acts_s1),torch.sum(acts_p1))
                elif i == 1:
                    prob = self.cp * p
                    imgs, mask2, acts, acts_s2, acts_p2, rand_index2 = op(imgs, original_imgs, acts, original_acts, original_labs, prob, args)
                    # print(torch.sum(acts_s2), torch.sum(acts_p2))
                    
                    
                    acts_1 = acts_s1 * mask2
                    # print(acts_1.shape)
                    acts_1 = torch.sum(acts_1, (1,2,3))
                    acts_2 = acts_p1 * mask2
                    acts_2 = torch.sum(acts_p1, (1,2,3))
                    acts_3 = torch.sum(acts_p2, (1,2,3))
                    
                    total_activation = acts_1 + acts_2 + acts_3
                    ratio1 = (acts_1 / total_activation)[:,None]
                    ratio2 = (acts_2 / total_activation)[:,None]
                    ratio3 = (acts_3 / total_activation)[:,None]
                    
                    labs1 = original_labs
                    labs2 = original_labs[rand_index1,:]
                    labs3 = original_labs[rand_index2,:]
                    
                    labs = labs1 * ratio1 + labs2 * ratio2 + labs3 * ratio3
            

        return imgs, labs
    