import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
from loguru import logger
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset

from dataloader import get_dataloader


class DTDP_dataset(object):
    def __init__(self, dataset_all,num_classes,epsilon,kappa,indices_init=None):
        self.dataset_all=dataset_all
        if indices_init==None:
            self.indices_init=list(range(len(dataset_all)))
        else:
            self.indices_init=indices_init
        self.indices=self.indices_init
        self.epsilon=epsilon
        self.kappa=kappa
        self.num_classes=num_classes
        self.score=np.zeros(len(self.indices_init))
        self.dataset_split()

    def get_dataset_init(self):
        return Subset(self.dataset_all,self.indices_init)     
        
    def get_dataset(self):
        return Subset(self.dataset_all,self.indices) 
        
    def dataset_split(self):
        self.indices_split=[[] for i in range(self.num_classes)]
        for index in self.indices_init:
            label=self.dataset_all[index][-1]
            self.indices_split[label].append(index)   
        
    def update_score(self, model,batch_size=64):
        dataset=self.get_dataset_init()
        model.eval()
        logger.info("updating score")
        dataloader_kwargs={"mode":'test',"batch_size":batch_size}
        dataloader=get_dataloader(dataset,**dataloader_kwargs)
        for i,data in enumerate(dataloader):
            index=data[0]
            input=data[1].cuda()
            label=data[2].cuda()
            prob=F.softmax(model(input),dim=-1).cpu().data.numpy()
            conf=np.max(prob,axis=-1)
            self.score[i*batch_size:(i+1)*batch_size]=self.kappa*conf+(1-self.kappa)*self.score[i*batch_size:(i+1)*batch_size]
            
            
    def update_indices(self):
        logger.info("updating indices")
        self.indices=[]
        for i in range(self.num_classes):
            class_score=self.score[self.indices_split[i]]
            y=np.argsort(class_score)
            num_cut=int(np.floor(len(y)*self.epsilon))
            y_cut=y[num_cut:]
            indices_left=list(np.array(self.indices_split[i])[y_cut])
            self.indices+=indices_left
        