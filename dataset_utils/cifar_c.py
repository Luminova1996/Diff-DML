import os
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as Data
import torch
import numpy as np


from .cifar import cifar10, cifar100 

CIFAR10_C_PATH="../data/CIFAR-10-C/"
CIFAR100_C_PATH="../data/CIFAR-100-C/"
CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

class CIFAR_C_Dataset(Data.Dataset):
    def __init__(self, dataset, transform, **kwargs):
        self.dataset=dataset 
        self.transform=transform       

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x = self.transform(self.dataset[index]['img'])
        y = self.dataset[index]['label']
        return index,x,y


class cifar10_C(cifar10):
    def __init__(self,noise_level,**kwargs):
        self.noise_level=noise_level
        super().__init__(**kwargs)

    def split_dataset(self,train,val,test): 
        self.dataset={'train':None,'val':None,'test':None}
        if test:  
            all_dataset=[]
            label_path=os.path.join(CIFAR10_C_PATH,'labels'+'.npy')
            label=np.load(label_path)
            for noise in CORRUPTIONS:
                data_path=os.path.join(CIFAR10_C_PATH,noise+'.npy')
                data=np.load(data_path)[(self.noise_level-1)*10000:self.noise_level*10000]
                for i in range(10000):
                    all_dataset.append({'img':data[i],'label':label[i]})


            self.dataset['test']=CIFAR_C_Dataset(dataset=all_dataset,transform=self.transform_test)



class cifar100_C(cifar100):
    def __init__(self,noise_level,**kwargs):
        self.noise_level=noise_level
        super().__init__(**kwargs)

    def split_dataset(self,train,val,test): 
        self.dataset={'train':None,'val':None,'test':None}
        if test:  
            all_dataset=[]
            label_path=os.path.join(CIFAR100_C_PATH,'labels'+'.npy')
            label=np.load(label_path)
            for noise in CORRUPTIONS:
                data_path=os.path.join(CIFAR100_C_PATH,noise+'.npy')
                data=np.load(data_path)[(self.noise_level-1)*10000:self.noise_level*10000]
                for i in range(10000):
                    all_dataset.append({'img':data[i],'label':label[i]})


            self.dataset['test']=CIFAR_C_Dataset(dataset=all_dataset,transform=self.transform_test)