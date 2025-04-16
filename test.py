import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
from loguru import logger
from tqdm import tqdm

from model import get_model
from align import Align_model
from opts import build_opts
from utils import *
from dataset_utils import get_dataset
from metric import ECE, AdaECE, Classwise_ECE, MCE, Full_ECE
from dataloader import get_dataloader

from scaling import Temperature,Vector,Dirichlet,fit_scaling_model


args=build_opts()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.scaling_method != 'None':
    logger.add(os.path.join(args.checkpoint_path,"test-{time}-%s.log"%(args.scaling_method)), rotation="500 KB")
else:
    logger.add(os.path.join(args.checkpoint_path,"test-{time}.log"), rotation="500 KB")


dataset_kwargs={
        'cutout':args.cutout,
        'init_model':args.tokenizer_name,
        'dataset':args.dataset,
        'dataset_part':args.dataset_part,        
        'train':0,
        'val':1,
        'test':1,
        'noise_level':args.noise_level,
}
dataset=get_dataset(**dataset_kwargs)
testset=dataset.dataset['test']

input_size=dataset.input_size
num_classes=dataset.num_classes

model_kwargs_A={
    'model_type':args.model_type_A,
    'input_size':input_size,
    'num_hidden_layers':args.num_hidden_layers_A,
    'hidden_size':args.hidden_size_A,
    'kernel_size':args.kernel_size_A,
    'hidden_channels':args.hidden_channels_A,
    'pretrained':args.pretrained_A,
    'num_classes':num_classes,
    'tokenizer':getattr(dataset, 'tokenizer', None),
    'config':getattr(dataset, 'config', None),
}
model_A=get_model(**model_kwargs_A).cuda()

model_A_load_from=getattr(args, 'model_A_load_from', None)

if model_A_load_from!=None:    
    load_state_dict(input=model_A,input_type='model_A',path=model_A_load_from)

def iter_printer(iterater):
    return tqdm(iterater,mininterval=5,maxinterval=100,miniters=5)

class LogitsDataset(torch.utils.data.Dataset):
    def __init__(self, logits, labels):
        super(LogitsDataset, self).__init__()
        self.logits = logits
        self.labels = labels
        
    def __len__(self):
        return self.logits.shape[0]

    def __getitem__(self, index):
        logits = self.logits[index, :].astype(np.float32)
        labels = self.labels[index].astype(np.long)
        return logits, labels

@logger.catch
def test(model):
    dataloader_kwargs={"mode":'test',"batch_size":args.batch_size}
    dataloader=get_dataloader(testset,**dataloader_kwargs)
    model.eval()
    logger.info(len(list(testset)))
    temperature=1
    if args.scaling_method != 'None':
        valset=dataset.dataset['val']
        val_dataloader_kwargs={"mode":'test',"batch_size":args.batch_size}
        val_dataloader=get_dataloader(valset,**dataloader_kwargs)

        pred_list_val=np.zeros((len(valset),num_classes))
        label_list_val=np.zeros(len(valset))
        for i,data in enumerate(iter_printer(val_dataloader)):
            index=data[0]
            input=data[1].cuda()
            label=data[2].cuda()
            pred=model(input).cpu().data.numpy()
            pred_list_val[i*args.batch_size:(i+1)*args.batch_size,:]=pred
            label_list_val[i*args.batch_size:(i+1)*args.batch_size]=label.cpu().data.numpy()
        
        dataset_logits=LogitsDataset(pred_list_val,label_list_val)
        dataloader_logits=get_dataloader(dataset_logits,**dataloader_kwargs)
        calibrator=fit_scaling_model(args.scaling_method,dataloader_logits,num_classes,binary_loss=False,regularization=False,num_epochs=200)
            
    test_count = 0
    
    prob_list=np.zeros((len(testset),num_classes))
    label_list=np.zeros(len(testset))

    for i,data in enumerate(iter_printer(dataloader)):
        index=data[0]
        input=data[1].cuda()
        label=data[2].cuda()
        logits=model(input)
        if args.scaling_method != 'None':
            logits=calibrator(logits)
        prob=F.softmax(logits,dim=-1).cpu().data.numpy()
        prob_classes=np.argmax(prob,axis=-1)
        test_count+=np.sum(prob_classes==label.cpu().data.numpy())

        label_list[i*args.batch_size:(i+1)*args.batch_size]=label.cpu().data.numpy()
        prob_list[i*args.batch_size:(i+1)*args.batch_size,:]=prob

    test_acc=test_count/len(testset)*100
    logger.info('model acc: %f'%(test_acc))
    ECE_out=ECE(prob_list,label_list,args.ECE_bin)*100
    MCE_out=MCE(prob_list,label_list,args.ECE_bin)*100
    AdaECE_out=AdaECE(prob_list,label_list,args.ECE_bin)*100
    Classwise_ECE_out=Classwise_ECE(prob_list,label_list,args.ECE_bin)*100
    Full_ECE_out=Full_ECE(prob_list,label_list,args.ECE_bin)*100
    logger.info('model ECE: %f, AdaECE:%f, ClasswiseECE:%f, MCE:%f, Full_ECE:%f'%(ECE_out,AdaECE_out,Classwise_ECE_out,MCE_out,Full_ECE_out))

test(model_A)   
logger.success("Done!")