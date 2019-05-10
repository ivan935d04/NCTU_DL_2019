from __future__ import print_function, division
import os 
import torch 
import pandas as pd 
import pickle
import numpy as np 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils 
from PIL import Image
from skimage import io, transform
import cv2


class AllData(Dataset):
    def __init__(self,root,mode='restore',name=None,transform=None):
        dirs = os.listdir(root)
        self.directorys = [os.path.join(root,k) for k in dirs]
        self.imgs = []
        self.name = name
        self.transform = transform

        if mode == "create":
            for number in range(len(self.directorys)):
                imgs = os.listdir(self.directorys[number])
                self.imgs += [os.path.join(self.directorys[number],img) for img in imgs]
                
            self.store()
        if mode == "restore":
            self.imgs=pickle.load(open(name,'rb'))

    def __getitem__(self,index):
        img_path = self.imgs[index]
        # print(img_path.split('/')[3])
        label =None

        if 'butterfly' in img_path.split('/')[3]:
            label = 0
        elif 'cat' in img_path.split('/')[3]:
            label = 1
        elif 'chicken' in img_path.split('/')[3]:
            label = 2
        elif 'cow' in img_path.split('/')[3]:
            label = 3
        elif  'dog' in img_path.split('/')[3]:
            label = 4
        elif  'elephant' in img_path.split('/')[3]:
            label = 5
        elif  'horse' in img_path.split('/')[3]:
            label = 6
        elif  'sheep' in img_path.split('/')[3]:
            label = 7
        elif  'spider' in img_path.split('/')[3]:
            label = 8
        elif  'squirrel' in img_path.split('/')[3]:
            label = 9
        
        pil_img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)

        sample = {'image':pil_img, 'label':label}
        if self.transform:
            sample = self.transform(sample)
        return sample

    
    def __len__(self):
        return len(self.imgs)

    def store(self):
        pickle.dump(self.imgs,open(self.name,"wb"))


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': label}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))


        return {'image': img, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': label}

    

        
        






