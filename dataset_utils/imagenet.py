import torch 
from torchvision import datasets
from torchvision import transforms
import numpy as np
from .cifar import Cutout

IMAGENET_TRAIN_PATH="../data/imagenet/train/"
IMAGENET_VAL_PATH="../data/imagenet/valset/"
IMAGENET_TEST_PATH="../data/imagenet/testset/"

IMAGENET_TRAIN_MEAN=(0.485, 0.456, 0.406)
IMAGENET_TRAIN_STD=(0.229, 0.224, 0.225)


class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return index, image, target

class ImageNet(object):
    def __init__(self,train=0,val=0,test=0,cutout=0,**kwargs):
        
        self.cut=cutout
        self.get_transforms()
        self.split_dataset(train=train,val=val,test=test)
        self.input_size=224*224*3
        self.num_classes=1000 
        self.SGD_lrdecay_gamma=0.1
        self.SGD_lrdecay_step=50
        self.SGD_lrdecay_begin=50
        
    def get_transforms(self):

        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)
        ])
          

        self.transform_test=transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD),
        ])
        
    def split_dataset(self,train,val,test):
    
        self.dataset={'train':None,'val':None,'test':None}
        if train:
            self.dataset['train'] = CustomImageFolder(root=IMAGENET_TRAIN_PATH, transform=self.transform_train) 
        
        if val:
            self.dataset['val'] = CustomImageFolder(root=IMAGENET_VAL_PATH, transform=self.transform_test) 
            
        if test:      
            self.dataset['test'] = CustomImageFolder(root=IMAGENET_TEST_PATH, transform=self.transform_test) 