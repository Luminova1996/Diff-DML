import torch 
from torchvision import datasets
from torchvision import transforms
import numpy as np
from .cifar import Cutout

TINY_IMAGENET_TRAIN_PATH="../data/tiny-imagenet-200/train/"
TINY_IMAGENET_VAL_PATH="../data/tiny-imagenet-200/valset/"
TINY_IMAGENET_TEST_PATH="../data/tiny-imagenet-200/testset/"

IMAGENET_TRAIN_MEAN=(0.485, 0.456, 0.406)
IMAGENET_TRAIN_STD=(0.229, 0.224, 0.225)

class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return index, image, target

class Tiny_ImageNet(object):
    def __init__(self,train=0,val=0,test=0,cutout=0,**kwargs):
        
        self.cut=cutout
        self.get_transforms()
        self.split_dataset(train=train,val=val,test=test)
        self.input_size=32*32*3
        self.num_classes=200 
        self.SGD_lrdecay_gamma=0.1
        self.SGD_lrdecay_step=50
        self.SGD_lrdecay_begin=50
        
    def get_transforms(self):

        self.transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)
        ])
          
        if self.cut:
            self.transform_test=transforms.Compose([
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD),
                Cutout(n_holes=1, length=16),
            ])              
        else:
            self.transform_test=transforms.Compose([
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD),
            ])
        
    def split_dataset(self,train,val,test):
    
        self.dataset={'train':None,'val':None,'test':None}
        if train:
            self.dataset['train'] = CustomImageFolder(root=TINY_IMAGENET_TRAIN_PATH, transform=self.transform_train) 
        
        if val:
            self.dataset['val'] = CustomImageFolder(root=TINY_IMAGENET_VAL_PATH, transform=self.transform_test) 
            
        if test:      
            self.dataset['test'] = CustomImageFolder(root=TINY_IMAGENET_TEST_PATH, transform=self.transform_test) 