from torchvision import datasets
from torchvision import transforms
import torch.utils.data as Data
import torch
import numpy as np

CIFAR10_TRAIN_MEAN=(0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD=(0.2023, 0.1994, 0.2010)

class CustomCIFAR10(datasets.CIFAR10):       
    
    def __getitem__(self, index):
        image, target = super().__getitem__(index)       
        return index, image, target

class cifar10(object):

    def __init__(self,train=0,val=0,test=0,cutout=0,**kwargs):
    
        self.cut=cutout
        self.get_transforms()
        self.split_dataset(train=train,val=val,test=test)
        self.input_size=32*32*3
        self.classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.num_classes=len(self.classes) 
        self.SGD_lrdecay_gamma=0.1
        self.SGD_lrdecay_step=50
        self.SGD_lrdecay_begin=50
        
    def get_transforms(self):

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
        ])
          
        if self.cut:
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
                Cutout(n_holes=1, length=16),
            ])

        else:
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
            ])
            
    def split_dataset(self,train,val,test):
    
        self.dataset={'train':None,'val':None,'test':None}
        if train:
            trainset_all=CustomCIFAR10('../data/',train=True,download=True,transform=self.transform_train)
            train_indices=list(range(45000))
            self.dataset['train']=Data.Subset(trainset_all,train_indices)
        
        if val:
            valset_all=CustomCIFAR10('../data/',train=True,download=True,transform=self.transform_test)        
            val_indices=list(range(45000,50000))
            self.dataset['val']=Data.Subset(valset_all,val_indices)  
        if test:      
            self.dataset['test']=CustomCIFAR10('../data/',train=False,download=True,transform=self.transform_test)


CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

class CustomCIFAR100(datasets.CIFAR100):       
    
    def __getitem__(self, index):
        image, target = super().__getitem__(index)       
        return index, image, target

class cifar100(object):
    def __init__(self,train=0,val=0,test=0,cutout=0,**kwargs):
     
        self.cut=cutout
        self.get_transforms()
        self.split_dataset(train=train,val=val,test=test)
        self.input_size=32*32*3
        self.classes=['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud',
                'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
                'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 
                'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 
                'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
                'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
                'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
                'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
                'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',
                'whale', 'willow_tree', 'wolf', 'woman', 'worm']  
        self.num_classes=len(self.classes)  
        self.SGD_lrdecay_gamma=0.1
        self.SGD_lrdecay_step=50
        self.SGD_lrdecay_begin=50
        
    def get_transforms(self):
    
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
        ])
          
        if self.cut:
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
                Cutout(n_holes=1, length=16),
            ])

        else:
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
            ])
            
    def split_dataset(self,train,val,test):
    
        self.dataset={'train':None,'val':None,'test':None}
        
        if train:
            trainset_all=CustomCIFAR100('../data/',train=True,download=True,transform=self.transform_train)
            train_indices=list(range(45000))
            self.dataset['train']=Data.Subset(trainset_all,train_indices)
        
        if val:
            valset_all=CustomCIFAR100('../data/',train=True,download=True,transform=self.transform_test)        
            val_indices=list(range(45000,50000))
            self.dataset['val']=Data.Subset(valset_all,val_indices)  
            
        if test:      
            self.dataset['test']=CustomCIFAR100('../data/',train=False,download=True,transform=self.transform_test)
 
class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
        