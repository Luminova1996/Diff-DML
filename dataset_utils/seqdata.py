import torch
import json
from loguru import logger
from transformers import AutoConfig, AutoTokenizer
import torch.utils.data as Data
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../')
from utils import *

IMDB_PATH="../data/IMDB/imdb-gpt2-large.pth"
SST2_PATH="../data/SST/sst-gpt2-large.pth"
AGNews_TRAIN_PATH="../data/AGNews/AGNews-train-gpt2-large.pth"
AGNews_TEST_PATH="../data/AGNews/AGNews-test-gpt2-large.pth"

class SeqDataset(Data.Dataset):
    def __init__(self, tokenizer, dataset_path, max_tokens=256, **kwargs):

        self.max_seq_length=max_tokens
        self.tokenizer=tokenizer
        self.tokenize_dataset(dataset_path)
        
    @logger.catch 
    @timmer 
    def tokenize_dataset(self,dataset_file):    
        logger.info('load data from %s'%(dataset_file))
        dataset_all=pickle_load_file(dataset_file)
        self.dataset=dataset_all
        #logger.info('tokenizing:')
        #self.dataset=[]
        #for data in tqdm(dataset_all,mininterval=5,maxinterval=100,miniters=5):
        #    self.dataset.append({'seq':self.tokenizer.encode(data['seq']), 'label':data['label']})
     
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x = padding([self.dataset[index]['seq']],max_len=self.max_seq_length,pads=self.tokenizer.eos_token_id,fill_to_max=True)[0]
        y = self.dataset[index]['label']
        return index,x,y
    
#    def collate_func(self, batch):
        #seq=padding([sample[0] for sample in batch],max_len=self.max_seq_length,pads=self.tokenizer.eos_token_id)
#        label=np.array([sample[1] for sample in batch])
#        seq=torch.from_numpy(seq).long()
#        label=torch.from_numpy(label).long()
#        data=[seq,label]

#        return data 

class IMDB(object):

    def __init__(self,init_model,train=0,val=0,test=0,**kwargs):
        
        self.init_model=init_model
        self.input_size=None
        self.classes=['negative','positive']
        self.num_classes=len(self.classes) 
        self.tokenizer=AutoTokenizer.from_pretrained(self.init_model)
        self.config=AutoConfig.from_pretrained(self.init_model)
        self.split_dataset(train=train,val=val,test=test)
        self.SGD_lrdecay_gamma=0.1
        self.SGD_lrdecay_step=20
        self.SGD_lrdecay_begin=30

    def split_dataset(self,train,val,test):
        dataset_all=SeqDataset(self.tokenizer,IMDB_PATH)
        self.max_seq_length=dataset_all.max_seq_length
        self.dataset={'train':None,'val':None,'test':None}
        if train:
            train_indices=list(range(35000))
            self.dataset['train']=Data.Subset(dataset_all,train_indices)
        
        if val:   
            val_indices=list(range(35000,40000))
            self.dataset['val']=Data.Subset(dataset_all,val_indices)
              
        if test: 
            test_indices=list(range(40000,50000))    
            self.dataset['test']=Data.Subset(dataset_all,test_indices)
       
class AGNews(object):

    def __init__(self,init_model,train=0,val=0,test=0,**kwargs):
        
        self.init_model=init_model
        self.input_size=None
        self.classes=['world','sports','business','sci/tech']
        self.num_classes=len(self.classes) 
        self.tokenizer=AutoTokenizer.from_pretrained(self.init_model)
        self.config=AutoConfig.from_pretrained(self.init_model)
        self.split_dataset(train=train,val=val,test=test)
        self.SGD_lrdecay_gamma=0.1
        self.SGD_lrdecay_step=20
        self.SGD_lrdecay_begin=30

    def split_dataset(self,train,val,test):
        
        self.dataset={'train':None,'val':None,'test':None}
        trainset_all=SeqDataset(self.tokenizer,AGNews_TRAIN_PATH)
        self.max_seq_length=trainset_all.max_seq_length
        if train:
            train_indices=list(range(110000))
            self.dataset['train']=Data.Subset(trainset_all,train_indices)
        
        if val:   
            val_indices=list(range(115000,120000))
            self.dataset['val']=Data.Subset(trainset_all,val_indices)
              
        if test: 
            testset_all=SeqDataset(self.tokenizer,AGNews_TEST_PATH)
            #test_indices=list(range(40000,50000))    
            self.dataset['test']=testset_all

class SST2(object):

    def __init__(self,init_model,train=0,val=0,test=0,**kwargs):
        
        self.init_model=init_model
        self.input_size=None
        self.classes=['negative','positive']
        self.num_classes=len(self.classes) 
        self.tokenizer=AutoTokenizer.from_pretrained(self.init_model)
        self.config=AutoConfig.from_pretrained(self.init_model)
        self.split_dataset(train=train,val=val,test=test)
        self.SGD_lrdecay_gamma=0.1
        self.SGD_lrdecay_step=20
        self.SGD_lrdecay_begin=30

    def split_dataset(self,train,val,test):
        self.dataset={'train':None,'val':None,'test':None}
        dataset_all=SeqDataset(self.tokenizer,SST2_PATH)
        self.max_seq_length=dataset_all.max_seq_length

        if val:   
            val_indices=list(range(0,1000))
            self.dataset['val']=Data.Subset(dataset_all,val_indices)

        if test:    
            self.dataset['test']=dataset_all
        
#        dataset_all=SeqDataset(self.tokenizer,IMDB_PATH)
#        self.dataset={'train':None,'val':None,'test':None}
#        if train:
#            train_indices=list(range(35000))
#            self.dataset['train']=Data.Subset(dataset_all,train_indices)
        
#        if val:   
#            val_indices=list(range(35000,40000))
#            self.dataset['val']=Data.Subset(dataset_all,val_indices)
              
#        if test: 
#            test_indices=list(range(40000,50000))    
#            self.dataset['test']=Data.Subset(dataset_all,test_indices)