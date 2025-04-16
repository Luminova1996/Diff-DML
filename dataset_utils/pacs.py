from torchvision import datasets
from torchvision import transforms
import torch.utils.data as Data
import torch
import numpy as np
from PIL import Image
import sys
sys.path.append('../')
from utils import *
from .cifar import Cutout

PACS_PATH="../data/PACS/pacs.pth"
IMAGENET_TRAIN_MEAN=(0.485, 0.456, 0.406)
IMAGENET_TRAIN_STD=(0.229, 0.224, 0.225)

class PACSDataset(Data.Dataset):
    def __init__(self, dataset, transform, **kwargs):

        self.dataset=dataset 
        self.transform=transform       

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x = self.transform(self.dataset[index]['img'])
        y = self.dataset[index]['label']
        return index,x,y

class PACS(object):

    def __init__(self,train=0,val=0,test=0,cutout=0,dataset_part=-1,**kwargs):
    
        self.cut=cutout
        self.dataset_part=dataset_part
        self.get_transforms()
        self.split_dataset(train=train,val=val,test=test)
        self.input_size=64*64*3
        self.domains=['art', 'sketch', 'cartoon', 'photo']
        self.classes=['house', 'dog', 'giraffe', 'elephant', 'horse', 'guitar', 'person']
        self.num_classes=len(self.classes) 
        self.SGD_lrdecay_gamma=0.1
        self.SGD_lrdecay_step=50
        self.SGD_lrdecay_begin=50
        
    def get_transforms(self):

        self.transform_train = transforms.Compose([
            Image.fromarray,
            transforms.Resize(64),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD),
        ])
          
        if self.cut:
            self.transform_test = transforms.Compose([
                Image.fromarray,
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD),
                Cutout(n_holes=1, length=32),
            ])

        else:
            self.transform_test = transforms.Compose([
                Image.fromarray,
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD),
            ])
            
    def split_dataset(self,train,val,test):
        
        dataset_all=pickle_load_file(PACS_PATH)
        self.dataset={'train':None,'val':None,'test':None}
        if train:
            self.dataset['train']=PACSDataset([x for x in dataset_all[1000:] if x['domain']!=self.dataset_part],transform=self.transform_train)
        
        if val:
            self.dataset['val']=PACSDataset([x for x in dataset_all[:1000] if x['domain']!=self.dataset_part],transform=self.transform_test)  
            
        if test:      
            self.dataset['test']=PACSDataset([x for x in dataset_all if x['domain']==self.dataset_part],transform=self.transform_test)