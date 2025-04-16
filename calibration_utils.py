import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from utils import *

ignore_index=-100

def base_loss(input,label,eps=0,focal=0,dual_focal=0,conf_pred=None,conf_pred_mode='kl-div'): #input:[batch_size, num_classes] label:[batch_size]
    if eps==0 and focal==0 and dual_focal==0 and (conf_pred==None):
        loss = F.nll_loss(input,label)
    elif conf_pred is not None:
        log_conf = take_index(input,label.unsqueeze(-1)).squeeze(-1)
        conf=torch.exp(log_conf)
        if conf_pred_mode=='kl-div':
            loss = -torch.sum(conf_pred*torch.log(conf+1e-10)+(1-conf_pred)*torch.log(1-conf+1e-10))/input.size(0)
        elif conf_pred_mode=='L1':
            loss = F.l1_loss(conf_pred,conf)
        else:
            raise ValueError("Unrecognizable conf_pred_mode type")
    elif eps>0:
        loss = smoothing_xe(input,label,eps)
    elif focal!=0:
        loss = focal_xe(input,label,focal) 
    elif dual_focal!=0:
        loss = dual_focal_xe(input,label,dual_focal)    
    else:
        raise ValueError("Unrecognizable loss type")
    return loss

def classification_loss(input,label,**kwargs): #input:[batch_size,num_classes] label:[batch_size]
    eps=kwargs.get('eps',0)
    focal=kwargs.get('focal',0)
    dual_focal=kwargs.get('dual_focal',0)
    huber_gamma=kwargs.get('huber_gamma',0)
    huber_alpha=kwargs.get('huber_alpha',0)
    MDCA_beta=kwargs.get('MDCA_beta',0)
    FDCA_beta=kwargs.get('FDCA_beta',0)
    FDCA_M=kwargs.get('FDCA_M',10)
    MMCE_lambda=kwargs.get('MMCE_lambda',0)
    DCA_beta=kwargs.get('DCA_beta',0)
    threshold_loss_beta=kwargs.get('threshold_loss_beta',0)
    p_threshold=kwargs.get('p_threshold',1.)
    conf_pred=kwargs.get('conf_pred',None)
    conf_pred_beta=kwargs.get('conf_pred_beta',0)
    conf_pred_mode=kwargs.get('conf_pred_mode','kl-div')
    use_ce=kwargs.get('use_ce',1)


    loss=0
    if use_ce:
        loss=base_loss(input,label,eps=eps,focal=focal,dual_focal=dual_focal)
    if conf_pred_beta!=0:
        #loss=(1-conf_pred_beta)*loss+conf_pred_beta*base_loss(input,label,eps=eps,focal=focal,conf_pred=conf_pred)
        loss=loss+conf_pred_beta*base_loss(input,label,eps=eps,focal=focal,conf_pred=conf_pred,conf_pred_mode=conf_pred_mode)

    if threshold_loss_beta!=0 and p_threshold==-1:        
        input_arg_index=torch.argmax(input,dim=-1)
        label_tmp=torch.clone(label)
        label_tmp[label_tmp==input_arg_index]=ignore_index
        input_tmp=input[label_tmp!=ignore_index,:]
        label_tmp=label_tmp[label_tmp!=ignore_index]
        
        loss=(1-threshold_loss_beta)*loss+threshold_loss_beta*base_loss(input_tmp,label_tmp,eps=eps,focal=focal)
    
    elif threshold_loss_beta>0 and p_threshold!=1:
        p_label=take_index(F.softmax(input,dim=-1),label.unsqueeze(-1)).squeeze(-1)
        label_tmp=torch.clone(label)
        label_tmp[p_label>p_threshold]=ignore_index
        input_tmp=input[label_tmp!=ignore_index,:]
        label_tmp=label_tmp[label_tmp!=ignore_index]
        
        loss=(1-threshold_loss_beta)*loss+threshold_loss_beta*base_loss(input_tmp,label_tmp,eps=eps,focal=focal)

    if huber_gamma!=0:
        loss_huber=Huber_loss(input, label, alpha=huber_alpha)
        loss+=huber_gamma*loss_huber
    if MDCA_beta!=0:
        loss+=MDCA_beta*MDCA(input,label)
    if FDCA_beta!=0:
        loss+=FDCA_beta*FDCA(input,label)
    if MMCE_lambda!=0:
        loss+=MMCE_lambda*MMCE(input,label)
    if DCA_beta!=0:
        loss+=DCA_beta*DCA(input,label)
    return loss       

def smoothing_xe(input,label,eps=0,**kwargs):
    num_classes=input.size(-1)
    one_hot_label=F.one_hot(label,num_classes).float()
    smoothed_label=one_hot_label*(1-eps)+eps/(num_classes-1)*(1-one_hot_label)
    return -1*torch.sum(input*smoothed_label)/input.size(0)

def focal_xe(input,label,focal=0,**kwargs):
    num_classes=input.size(-1)
    prob=F.softmax(input,dim=-1)
    input_target=take_index(input,label.unsqueeze(-1)).view(-1)
    prob_target=take_index(prob,label.unsqueeze(-1),fill_with=1).view(-1)
    if focal>=0:
        gamma=focal
    else:
        gamma=(prob_target.detach()<0.2)*2.0+3.0
    focal_loss=(1-prob_target).pow(gamma)*input_target
    return -1*torch.sum(focal_loss)/torch.sum(label!=ignore_index).type_as(input)

def dual_focal_xe(input,label,focal=0,**kwargs):
    target = label.view(-1,1)
    logp_k = F.log_softmax(input, dim=1)
    softmax_logits = logp_k.exp()
    logp_k = logp_k.gather(1, target)
    logp_k = logp_k.view(-1)
    p_k = logp_k.exp()  # p_k: probility at target label
    p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
    p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

    loss = -1 * (1 - p_k + p_j) ** focal * logp_k
    return loss.mean()
    
def logitnorm(input):
    output = input/torch.sqrt(torch.sum(input*input,dim=-1))[:,None]
    return output

def Huber_fun(x, alpha):
    if x<=alpha and x>=-alpha:
        y=0.5*x*x
    else:
        y=alpha*torch.abs(x-0.5*alpha)
    return y

def Huber_loss(input,label,alpha):
    batch_size=input.size(0)
    prob=F.softmax(input,dim=-1)
    conf, indice=torch.max(prob,dim=-1)
    a=torch.sum(indice==label)
    c=torch.sum(conf)
    loss=Huber_fun((c-a)/batch_size,alpha)
    return loss
    
def MDCA(input, label):
    batch_size=input.size(0)
    K=input.size(-1)
    prob=F.softmax(input,dim=-1)
    label_one_hot=F.one_hot(label,num_classes=K)
    loss=torch.sum(torch.abs(torch.sum(prob,dim=0)-torch.sum(label_one_hot,dim=0).float()))/(batch_size*K)
    return loss
    
def FDCA(input, label, M=10):
    batch_size=input.size(0)
    K=input.size(-1)
    prob=F.softmax(input,dim=-1)

    count_A=torch.zeros(M,device=input.device)
    count_C=torch.zeros(M,device=input.device)

    prob_bin=(prob*M).int() #[batch_size, num_classes]
    prob_bin[prob_bin==M]=M-1

    for k in range(M):
        count_A[k]+=torch.sum(F.one_hot(label,num_classes=K)*(prob_bin==k))
        count_C[k]+=torch.sum(prob*(prob_bin==k))

    loss=torch.sum(torch.abs(count_A[k]-count_C[k]))/batch_size
    return loss

def DCA(input, label):
    batch_size=input.size(0)
    prob=F.softmax(input,dim=-1)
    conf, indices=torch.max(prob,dim=-1)
    loss=torch.abs(torch.sum(conf)-torch.sum(indices==label))/batch_size
    return loss

def MMCE_kernel(x1,x2):
    y=torch.exp(-torch.abs(x1-x2)/0.4)
    return y

def MMCE(input, label):
    prob=F.softmax(input,dim=-1)
    conf, indices = torch.max(prob,dim=-1)
    m=torch.sum(indices==label)
    n=input.size(0)
    k=input.size(-1)
    correct=(indices==label).float()
    a=torch.zeros_like(correct)
    a+=correct*(1-conf)/n
    if n!=m:
        a+=(1-correct)*conf/(n-m)    

    q1=a.unsqueeze(-1).expand(-1,n)
    q2=torch.transpose(q1,-1,-2)
    q=q1*q2

    d1=conf.unsqueeze(-1).expand(-1,n)
    d2=torch.transpose(d1,-1,-2)
    d=MMCE_kernel(d1,d2)

    loss=torch.sqrt(torch.sum(q*d))
    return loss
    

