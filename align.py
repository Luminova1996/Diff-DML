import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from calibration_utils import classification_loss
from utils import *

class Align_model(nn.Module):
    def __init__(self, model_A, model_B_list, model_conf,align_flag=True):
        super().__init__()
        self.model_A=model_A
        self.model_B_list=nn.ModuleList(model_B_list)
        self.model_conf=model_conf
        self.num_B=len(self.model_B_list)
        self.align_flag=align_flag
       
    def get_loss(self, input, label,**kwargs):                
        use_ce=kwargs.get('use_ce',1)
        use_conf_model=kwargs.get('use_conf_model',0)
        tar_conf=kwargs.get('tar_conf',None)
        conf_pred_beta=kwargs.get('conf_pred_beta',0.)
        self_align=kwargs.get('self_align',0)
        self_align_delta=kwargs.get('self_align_delta',0.)
        smoothing_eps=kwargs.get('smoothing_eps',0)
        norm=kwargs.get('norm',0)
        fix_A=kwargs.get('fix_A',0)
        fix_B=kwargs.get('fix_B',0)
        dual_focal=kwargs.get('dual_focal',0)
        focal=kwargs.get('focal',0)
        train_B=kwargs.get('train_B',0)
        align_weight=kwargs.get('align_weight',1)
        align_weight_B=kwargs.get('align_weight_B',1)
        huber_gamma=kwargs.get('huber_gamma',0.)
        huber_alpha=kwargs.get('huber_alpha',0.)
        MDCA_beta=kwargs.get('MDCA_beta',0.)
        FDCA_beta=kwargs.get('FDCA_beta',0.)
        FDCA_M=kwargs.get('FDCA_M',10)
        MMCE_lambda=kwargs.get('MMCE_lambda',0.)
        DCA_beta=kwargs.get('DCA_beta',0.)
        fudge=kwargs.get('fudge',0)
        threshold_loss_beta=kwargs.get('threshold_loss_beta',0.)
        p_threshold=kwargs.get('p_threshold',1.)
        conf_pred_mode=kwargs.get('conf_pred_mode','kl-div')
        
        loss=0
        out_A=self.model_A.forward(input)
        pred_A=F.log_softmax(out_A,dim=-1) #[batchsize, num_classes] fudge:[batch_size, seq_len, num_classes]

        pred_A=pred_A.view(-1,pred_A.size(-1))
  
        pred_A_tmp=pred_A[label!=ignore_index,:]
        label_tmp=label[label!=ignore_index]   

        if norm:
            pred_A=logitnorm(pred_A)      
        prob_A_tmp=F.softmax(pred_A_tmp,dim=-1).detach()

        if tar_conf is not None:
            conf_A_pred_tmp=tar_conf[label!=ignore_index]
            loss_conf=0
        elif use_conf_model:
            conf_A_pred=F.sigmoid(self.model_conf(input)).squeeze(-1) #[batchsize]
            conf_A_pred=conf_A_pred.view(-1)
            conf_A_pred=conf_A_pred[label!=ignore_index]
            conf_label=(torch.argmax(pred_A_tmp,dim=-1)==label_tmp).float()
            loss_conf=-torch.sum((conf_label)*torch.log(conf_A_pred+1e-10)+(1-conf_label)*torch.log(1-conf_A_pred+1e-10))/conf_A_pred.size(0)
            loss+=loss_conf
            conf_A_pred_tmp=conf_A_pred.detach()
        else:
            conf_A_pred_tmp=None
            loss_conf=0
        
        classification_loss_kwargs={
            'eps':smoothing_eps,
            'dual_focal':dual_focal,
            'focal':focal,
            'huber_gamma':huber_gamma,
            'huber_alpha':huber_alpha,
            'MDCA_beta':MDCA_beta,
            'FDCA_beta':FDCA_beta,
            'FDCA_M':FDCA_M,
            'MMCE_lambda':MMCE_lambda,
            'DCA_beta':DCA_beta,
            'threshold_loss_beta':threshold_loss_beta,
            'p_threshold':p_threshold,
            'conf_pred':conf_A_pred_tmp,
            'conf_pred_beta':conf_pred_beta,
            'use_ce':use_ce,
            'conf_pred_mode':conf_pred_mode,
        }

        loss_xe=classification_loss(pred_A_tmp,label_tmp,**classification_loss_kwargs)
        
        if self_align:
            pred_A_tmp_=torch.normal(mean=pred_A_tmp,std=self_align_delta)
            prob_A_tmp_=F.softmax(pred_A_tmp_,dim=-1).detach()
            loss_self_align=torch.sum(prob_A_tmp_*(pred_A_tmp_.detach()-pred_A_tmp))/pred_A_tmp.size(0)
            loss_xe+=align_weight*loss_self_align
        if self.align_flag==True:
            pred_B_list=[]
            pred_B_tmp_list=[]
            prob_B_tmp_list=[]
            for model_B in self.model_B_list:
                pred_B=F.log_softmax(model_B(input),dim=-1)
                if fudge:
                    pred_B=pred_B.view(-1,pred_B.size(-1))
                    pred_B_tmp=pred_B[label!=ignore_index,:]
                else:
                    pred_B_tmp=pred_B

                prob_B_tmp=F.softmax(pred_B_tmp,dim=-1).detach()
                pred_B_list.append(pred_B)
                pred_B_tmp_list.append(pred_B_tmp)
                prob_B_tmp_list.append(prob_B_tmp)

            if train_B:
                for i in range(self.num_B):
                    loss_xe+=train_B*classification_loss(pred_B_tmp_list[i],label_tmp,**classification_loss_kwargs)
              
            if fix_A:
              pred_A_tmp=pred_A_tmp.detach()
            if fix_B:
                for i in range(self.num_B):
                    pred_B_tmp_list[i]=pred_B_tmp_list[i].detach()

            loss_align_A=0
            loss_align_B=0
            for i in range(self.num_B):
                loss_align_A+=torch.sum(prob_B_tmp_list[i]*(pred_B_tmp_list[i].detach()-pred_A_tmp))/pred_A_tmp.size(0)
                loss_align_B+=torch.sum(prob_A_tmp*(pred_A_tmp.detach()-pred_B_tmp_list[i]))/pred_A_tmp.size(0)
            loss_align_A/=self.num_B
            loss_align=loss_align_A*align_weight+loss_align_B*align_weight_B
            
            loss+=loss_xe+loss_align

        else:
            loss+=loss_xe    

        prob_out_A=F.softmax(pred_A,dim=-1).cpu().data.numpy()
        if self.align_flag==True:
            prob_out_B_list=[F.softmax(pred_B,dim=-1).cpu().data.numpy() for pred_B in pred_B_list]
            loss_align_out_A=loss_align_A.cpu().data.numpy()
            loss_align_out_B=loss_align_B.cpu().data.numpy()/self.num_B
        else:
            prob_out_B_list=None
            loss_align_out_A=0.
            loss_align_out_B=0.
        
        if use_conf_model:
            loss_conf_out=loss_conf.item()
            conf_A_pred_out=conf_A_pred.cpu().data.numpy()
        else:
            loss_conf_out=0.
            conf_A_pred_out=0.

        return loss,prob_out_A,prob_out_B_list,loss_align_out_A,loss_align_out_B,loss_conf_out,conf_A_pred_out











