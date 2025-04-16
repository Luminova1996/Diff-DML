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
from metric import ECE, AdaECE, Classwise_ECE, MCE
from dataloader import get_dataloader
from calibration_utils import get_temperature



args=build_opts()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

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

ensemble_models_path=[]
ensemble_models_path.append("log/log-cifar100-ensemble/single-SGD-part-0-resnet34-smoothing-0.0-focal-0.0-MDCA-0.0-MMCE-0.0-DCA-0.0-seed-1111/model1.best.th")
ensemble_models_path.append("log/log-cifar100-ensemble/single-SGD-part-0-resnet34-smoothing-0.0-focal-0.0-MDCA-0.0-MMCE-0.0-DCA-0.0-seed-2222/model1.best.th")
ensemble_models_path.append("log/log-cifar100-ensemble/single-SGD-part-0-resnet34-smoothing-0.0-focal-0.0-MDCA-0.0-MMCE-0.0-DCA-0.0-seed-5555/model1.best.th")
ensemble_models=[]

for i in range(len(ensemble_models_path)):
    model_kwargs={
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
    ensemble_models.append(get_model(**model_kwargs).cuda())
    load_state_dict(input=ensemble_models[i],input_type='model%d'%(i+1),path=ensemble_models_path[i])
 

def iter_printer(iterater):
    return tqdm(iterater,mininterval=5,maxinterval=100,miniters=5)

@logger.catch
def test():
    dataloader_kwargs={"mode":'test',"batch_size":args.batch_size}
    dataloader=get_dataloader(testset,**dataloader_kwargs)
    for model in ensemble_models:
        model.eval()
    logger.info(len(list(testset)))
    temperature=1
    if args.use_TS:
        valset=dataset.dataset['val']
        val_dataloader_kwargs={"mode":'test',"batch_size":args.batch_size}
        val_dataloader=get_dataloader(valset,**dataloader_kwargs)
        temperature=get_temperature(model, val_dataloader)
        

    test_count = 0
    
    prob_list=np.zeros((len(testset),num_classes))
    label_list=np.zeros(len(testset))

    for i,data in enumerate(iter_printer(dataloader)):
        index=data[0]
        input=data[1].cuda()
        label=data[2].cuda()
        prob=0
        for model in ensemble_models:
            prob+=(F.softmax(model(input)/temperature,dim=-1).cpu().data.numpy())
        prob/=len(ensemble_models)
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
    logger.info('model1 ECE: %f, AdaECE:%f, ClasswiseECE:%f, MCE:%f'%(ECE_out,AdaECE_out,Classwise_ECE_out,MCE_out))

test()   
logger.success("Done!")