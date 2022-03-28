import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from skimage import io
import numpy as np
import torchvision.transforms as transforms
from randaugment import RandAugment
ra = RandAugment(2,14)

class data_with_feature(Dataset):

    def __init__(self, root_dir, is_train, transform=None, ex_transform=None, rand_aug = False):

        self.ex_transform = ex_transform
        self.root_dir = root_dir
        self.transform = transform
        self.rand_aug = rand_aug

        if is_train:
            self.data_part = 'train'
        else:
            self.data_part = 'val'

        with open(os.path.join(self.root_dir, self.data_part + '.txt')) as file:
            list = file.readlines()
            self.lists = [i.rstrip() for i in list]

        with open(os.path.join(self.root_dir, self.data_part + 'f.txt')) as feature_file:
            feature_list = feature_file.readlines()
            self.feature_lists = [i.rstrip() for i in feature_list]

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.lists[idx])

        image = Image.fromarray(np.load(img_name))

        #print(image.ndim)
        #if image.ndim != 3:
        #    image = np.repeat(image[:,:, None],3,axis=2)

        feature_name = os.path.join(self.feature_lists[idx])
        feature_image = torch.tensor(io.imread(feature_name))
        
        if not self.rand_aug:
        
            if self.transform:
                
                image = self.transform(image)
                
                if self.ex_transform:
                    combined = torch.cat((image, feature_image[None,:,:]), 0)
                    combined = self.ex_transform(combined)
                    image = combined[:3,:,:]
                    feature_image = combined[3,:,:]
                    feature_image = torch.unsqueeze(feature_image, 0)
        
        else:
            
            # print(image)
            
            image, feature_image = ra(image,feature_image)
            
            # print(image)
            # assert False
            
            
            if self.transform:
                
                image = self.transform(image)
                
                if self.ex_transform:
                    combined = torch.cat((image, feature_image[None,:,:]), 0)
                    combined = self.ex_transform(combined)
                    image = combined[:3,:,:]
                    feature_image = combined[3,:,:]
                    feature_image = torch.unsqueeze(feature_image, 0)
            
            
        label = int(os.path.basename(os.path.dirname(self.lists[idx])))

        return image, feature_image, label