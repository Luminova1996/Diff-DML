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

from model import get_model
from align import Align_model
from opts import build_opts
from utils import *
from dataset_utils import get_dataset
from metric import ECE, AdaECE, Classwise_ECE, MCE, Full_ECE
from dataloader import get_dataloader
from data_pruning import DTDP_dataset


args=build_opts()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

tb_summary_writer = SummaryWriter(args.checkpoint_path)
logger.add(os.path.join(args.checkpoint_path,"train-{time}.log"), rotation="500 KB")


dataset_kwargs={
        'init_model':args.tokenizer_name,
        'dataset':args.dataset,
        'dataset_part':args.dataset_part,
        'train':1,
        'val':1,
        'test':0,
        'cutout':args.cutout,
}
dataset=get_dataset(**dataset_kwargs)
trainset=dataset.dataset['train']
testset=dataset.dataset['val']

input_size=dataset.input_size
num_classes=dataset.num_classes

if args.DTDP:
    logger.info('using DTDP dataset')
    DTDP=DTDP_dataset(trainset,num_classes=num_classes,epsilon=0.1,kappa=0.3)

model_kwargs_A={
    'model_type':args.model_type_A,
    'init_model':args.init_model_A,
    'input_size':input_size,
    'num_hidden_layers':args.num_hidden_layers_A,
    'hidden_size':args.hidden_size_A,
    'kernel_size':args.kernel_size_A,
    'hidden_channels':args.hidden_channels_A,
    'pretrained':args.pretrained_A,
    'num_classes':num_classes,
    'fudge':args.fudge,
    'tokenizer':getattr(dataset, 'tokenizer', None),
    'config':getattr(dataset, 'config', None),
}
model_A=get_model(**model_kwargs_A).cuda()

model_A_load_from=getattr(args, 'model_A_load_from', None)
infos_load_from=getattr(args, 'infos_load_from', None)
if model_A_load_from!=None:    
    load_state_dict(input=model_A,input_type='model_A',path=model_A_load_from)
if infos_load_from!=None:  
    infos=pickle_load_file(infos_load_from)

if args.align_flag:
    model_kwargs_B={
        'model_type':args.model_type_B,
        'init_model':args.init_model_B,
        'input_size':input_size,
        'num_hidden_layers':args.num_hidden_layers_B,
        'hidden_size':args.hidden_size_B,
        'kernel_size':args.kernel_size_B,
        'hidden_channels':args.hidden_channels_B,
        'pretrained':args.pretrained_B,
        'num_classes':num_classes,
        'fudge':args.fudge,
        'lora_rank':args.lora_rank,
        'tokenizer':getattr(dataset, 'tokenizer', None),
        'config':getattr(dataset, 'config', None),
        'random_init':args.random_init,
    }
    model_B_list=[get_model(**model_kwargs_B).cuda() for i in range(args.num_B)]

else:
    model_B_list=None

if args.use_conf_model:
    model_kwargs_conf={
        'model_type':args.model_type_conf,
        'init_model':args.init_model_conf,
        'input_size':input_size,
        'num_hidden_layers':args.num_hidden_layers_conf,
        'hidden_size':args.hidden_size_conf,
        'kernel_size':args.kernel_size_conf,
        'hidden_channels':args.hidden_channels_conf,
        'pretrained':args.pretrained_conf,
        'num_classes':1,
        'fudge':args.fudge,
        'tokenizer':getattr(dataset, 'tokenizer', None),
        'config':getattr(dataset, 'config', None),
    }
    model_conf=get_model(**model_kwargs_conf).cuda()
else:
    model_conf=None

align_model=Align_model(model_A,model_B_list,model_conf,args.align_flag).cuda()

if args.optimizer=='AdamW':
    optimizer_A = torch.optim.AdamW(align_model.model_A.parameters(),
                                  weight_decay=0.01,
                                  lr=args.lr_A)
    scheduler_A = torch.optim.lr_scheduler.StepLR(optimizer_A,step_size=50,gamma=0.5)
elif args.optimizer=='SGD':
    optimizer_A = torch.optim.SGD(align_model.model_A.parameters(),
                                  momentum=0.9, 
                                  weight_decay=5e-4,
                                  lr=args.lr_A)
    scheduler_A = torch.optim.lr_scheduler.StepLR(optimizer_A,step_size=dataset.SGD_lrdecay_step,gamma=dataset.SGD_lrdecay_gamma)
else:    
    raise ValueError("Unrecognizable optimizer type")    
    
if args.align_flag:
    if args.optimizer=='AdamW':        
        optimizer_B = torch.optim.AdamW(align_model.model_B_list.parameters(),
                                weight_decay=0.01,
                                lr=args.lr_B)
    elif args.optimizer=='SGD':
        optimizer_B = torch.optim.SGD(align_model.model_B_list.parameters(),
                                  momentum=0.9, 
                                  weight_decay=5e-4,
                                  lr=args.lr_B)
 
        if args.lr_B_decay:
            scheduler_B = torch.optim.lr_scheduler.StepLR(optimizer_B,step_size=dataset.SGD_lrdecay_step,gamma=dataset.SGD_lrdecay_gamma)

if args.use_conf_model:
    if args.optimizer=='AdamW':        
        optimizer_conf = torch.optim.AdamW(align_model.model_conf.parameters(),
                                weight_decay=0.01,
                                lr=args.lr_conf)
    elif args.optimizer=='SGD':
        optimizer_conf = torch.optim.SGD(align_model.model_conf.parameters(),
                                  momentum=0.9, 
                                  weight_decay=5e-4,
                                  lr=args.lr_conf) 
        if args.lr_conf_decay:
            scheduler_conf = torch.optim.lr_scheduler.StepLR(optimizer_conf,step_size=dataset.SGD_lrdecay_step,gamma=dataset.SGD_lrdecay_gamma)
    


def init_infos():
    infos={}
    infos['epoch']=0
    infos['step']=0
    infos['best_acc']=0.
    infos['overall_acc']=0.,
    infos['overall_acc-ECE']=-1e8
    infos['stage2']=0
    return infos

infos=init_infos()




if args.count_sample_acc:
    if args.count_sample_acc_len==-1:
        if args.count_acc_load_from is not None:
            train_sample_count=pickle_load_file(args.count_acc_load_from)
        else:
            train_sample_count=np.zeros((len(trainset),2))
    elif args.count_sample_acc_len>0:
        train_sample_count_memory=np.zeros((args.count_sample_acc_len,len(trainset),2))

def iter_printer(iterater,epoch):
    return tqdm(iterater, desc='epoch {}'.format(epoch),mininterval=5,maxinterval=100,miniters=5)

@logger.catch
def train_one_epoch(epoch):
    logger.info("training epoch %d"%(epoch))
    dataloader_kwargs={"mode":'train',"batch_size":args.batch_size}
    if args.DTDP:
        trainset_tmp=DTDP.get_dataset()
    else:
        trainset_tmp=trainset
    dataloader=get_dataloader(trainset_tmp,**dataloader_kwargs)
    align_model.train()

    train_loss = 0.0
    train_kl_A=0.0
    train_kl_B=0.0
    train_loss_conf=0.0

    if epoch>-1:
        self_align=args.self_align
    else:
        self_align=0
    if epoch>=args.use_conf_model_after:
        use_conf_model=args.use_conf_model
    else:
        use_conf_model=0

    if args.use_ce_max_epoch>=0 and epoch>=args.use_ce_max_epoch:
        use_ce=0
    else:
        use_ce=1

    for i,data in enumerate(iter_printer(dataloader,epoch)):
        index=data[0] 
        input=data[1].cuda()
        label=data[2].cuda()

        if args.fudge:
            index=index.unsqueeze(1).expand(input.size(0),input.size(1))
            index=index.reshape(-1)

            label=label.unsqueeze(1).expand(input.size(0),input.size(1))
            label=label.reshape(-1)
            mask=(input!=dataset.tokenizer.eos_token_id).type_as(input).view(-1)
            label[mask==0]=ignore_index #ignore masked tokens
        
        if args.count_sample_acc and epoch>=args.use_conf_model_after:
            if args.count_sample_acc_len>0:
                train_sample_count_tmp=np.sum(train_sample_count_memory,axis=0)
            else:
                train_sample_count_tmp=train_sample_count
            tar_conf=torch.tensor([train_sample_count_tmp[j][1]/train_sample_count_tmp[j][0] for j in index],device=input.device)
        else:
            tar_conf=None
            
        loss_kwargs={
            'smoothing_eps':args.smoothing_eps,
            'norm':args.norm_flag,
            'fix_A':args.fix_A,
            'fix_B':args.fix_B,
            'dual_focal':args.dual_focal,
            'focal':args.focal,
            'train_B':args.train_B,
            'self_align':args.self_align,
            'align_weight':args.align_weight,
            'align_weight_B':args.align_weight_B,
            'huber_gamma':args.huber_gamma,
            'huber_alpha':args.huber_alpha,
            'MDCA_beta':args.MDCA_beta,
            'FDCA_beta':args.FDCA_beta,
            'FDCA_M':args.FDCA_M,
            'MMCE_lambda':args.MMCE_lambda,
            'DCA_beta':args.DCA_beta,  
            'fudge':args.fudge, 
            'p_threshold':args.p_threshold,
            'threshold_loss_beta':args.threshold_loss_beta, 
            'conf_pred_beta':args.conf_pred_beta,
            'use_conf_model':use_conf_model, 
            'tar_conf':tar_conf,
            'use_ce':use_ce,
            'conf_pred_mode':args.conf_pred_mode,
        }
  
        loss,prob_A,prob_B_list,kl_A,kl_B,loss_conf,pred_conf=align_model.get_loss(input,label,**loss_kwargs)
        loss.backward()
        train_loss+=loss.item()
        train_kl_A+=kl_A
        train_kl_B+=kl_B
        train_loss_conf+=loss_conf
        
        #if args.optimizer=='AdamW':
        #    torch.nn.utils.clip_grad_norm_(align_model.parameters(), args.max_grad_norm)
        if args.count_sample_acc and epoch>=args.train_A_after and epoch<args.stop_train_conf_after:
            prob_A_classes=np.argmax(prob_A,axis=-1)
            class_match=(prob_A_classes==label.cpu().data.numpy())
            for j in range(len(index)):
                if label[j]!=ignore_index:
                    if args.count_sample_acc_len==-1:
                        train_sample_count[index[j]][0]+=1
                        train_sample_count[index[j]][1]+=class_match[j]
                    elif args.count_sample_acc_len>0:
                        train_sample_count_memory[epoch%args.count_sample_acc_len][index[j]][0]=1
                        train_sample_count_memory[epoch%args.count_sample_acc_len][index[j]][1]=class_match[j]

        if epoch<args.train_A_after and epoch>=args.use_conf_model_after:
            optimizer_A.zero_grad()
        else:
            optimizer_A.step()
            optimizer_A.zero_grad()
        if args.align_flag:
            optimizer_B.step()
            optimizer_B.zero_grad()
        if use_conf_model:
            if args.stop_train_conf_after>=0 and epoch<args.stop_train_conf_after:
                optimizer_conf.step()
            optimizer_conf.zero_grad()    
        infos['step']+=1
        
        if (i+1)%args.tb_record_every==0:
            tb_summary_writer.add_scalar('train_loss', train_loss/args.tb_record_every,infos['step'])
            tb_summary_writer.add_scalar('train_kl_A', train_kl_A/args.tb_record_every,infos['step'])
            tb_summary_writer.add_scalar('train_kl_B', train_kl_B/args.tb_record_every,infos['step'])
            tb_summary_writer.add_scalar('train_loss_conf', train_loss_conf/args.tb_record_every,infos['step'])
            train_loss = 0.0
            train_kl_A = 0.0
            train_kl_B = 0.0
            train_loss_conf = 0.0
    
    if args.optimizer=='SGD':
        schedule_begin=dataset.SGD_lrdecay_begin
    else:
        schedule_begin=0
    if epoch>=schedule_begin:
        scheduler_A.step()
        if args.align_flag and args.lr_B_decay:
            scheduler_B.step()
        if args.use_conf_model and args.lr_conf_decay:
            scheduler_conf.step()
        
    if args.DTDP and (epoch+1)%5==0:
        DTDP.update_score(model_A,args.batch_size)
        DTDP.update_indices()

@logger.catch
@torch.no_grad()
def eval_one_epoch(epoch):
    logger.info("testing epoch %d"%(epoch))
    dataloader_kwargs={"mode":'test',"batch_size":args.batch_size}
    dataloader=get_dataloader(testset,**dataloader_kwargs)
    align_model.eval()

    test_loss = 0.0
    test_count_A = 0
    test_count_B_list = np.zeros(args.num_B)
    loss_align = 0
    test_kl_A=0.0
    test_kl_B=0.0
    test_loss_conf=0.0
    test_conf_L1=0.0

    if args.fudge:
        mask_count=0
        prob_A_list=np.zeros((len(testset)*dataset.max_seq_length,num_classes))
        prob_B_list_list=[np.zeros((len(testset)*dataset.max_seq_length,num_classes)) for i in range(args.num_B)]
        label_list=np.zeros(len(testset)*dataset.max_seq_length)
    else:
        prob_A_list=np.zeros((len(testset),num_classes))
        prob_B_list_list=[np.zeros((len(testset),num_classes)) for i in range(args.num_B)]
        label_list=np.zeros(len(testset))

    if epoch>=args.use_conf_model_after:
        use_conf_model=args.use_conf_model
    else:
        use_conf_model=0

    for i,data in enumerate(iter_printer(dataloader,epoch)):
        index=data[0]
        input=data[1].cuda()
        label=data[2].cuda()
        if args.fudge:         
            label=label.unsqueeze(1).expand(input.size(0),input.size(1))
            label=label.reshape(-1)
            mask=(input!=dataset.tokenizer.eos_token_id).type_as(input)
            mask=mask.reshape(-1)
            label[mask==0]=ignore_index #ignore masked tokens
            mask_count+=torch.sum(mask)

        loss_kwargs={
            'norm':args.norm_flag,
            'align_weight':args.align_weight,
            'fudge':args.fudge,        
        }

        loss,prob_A,prob_B_list,kl_A,kl_B,loss_conf,conf_pred=align_model.get_loss(input,label,**loss_kwargs)
        test_loss+=loss.item()

        prob_A_classes=np.argmax(prob_A,axis=-1)
        test_count_A+=np.sum(prob_A_classes==label.cpu().data.numpy())
        test_loss_conf+=loss_conf
        conf_A=np.max(prob_A)
        test_conf_L1+=np.sum(np.abs(conf_A-conf_pred))


        if args.fudge:
            label_list[i*args.batch_size*dataset.max_seq_length:(i+1)*args.batch_size*dataset.max_seq_length]=label.cpu().data.numpy()
            prob_A_list[i*args.batch_size*dataset.max_seq_length:(i+1)*args.batch_size*dataset.max_seq_length,:]=prob_A
        else:
            label_list[i*args.batch_size:(i+1)*args.batch_size]=label.cpu().data.numpy()
            prob_A_list[i*args.batch_size:(i+1)*args.batch_size,:]=prob_A

        if args.align_flag==True:
            for j,prob_B in enumerate(prob_B_list):
                prob_B_classes=np.argmax(prob_B,axis=-1)
                test_count_B_list[j]+=np.sum(prob_B_classes==label.cpu().data.numpy())
                if args.fudge:
                    prob_B_list_list[j][i*args.batch_size*dataset.max_seq_length:(i+1)*args.batch_size*dataset.max_seq_length,:]=prob_B
                else:
                    prob_B_list_list[j][i*args.batch_size:(i+1)*args.batch_size,:]=prob_B
            test_kl_A+=kl_A
            test_kl_B+=kl_B
    if args.fudge:
        test_loss/=mask_count
        test_loss_conf/=mask_count
        test_acc_A=test_count_A/mask_count*100
        test_conf_L1/=mask_count
    else:
        test_loss/=len(testset)
        test_loss_conf/=len(testset)
        test_acc_A=test_count_A/len(testset)*100
        test_conf_L1/=len(testset)

    ECE_out_A=ECE(prob_A_list,label_list,args.ECE_bin)*100
    AdaECE_out_A=AdaECE(prob_A_list,label_list,args.ECE_bin)*100
    Classwise_ECE_out_A=Classwise_ECE(prob_A_list,label_list,args.ECE_bin)*100
    Full_ECE_out_A=Full_ECE(prob_A_list,label_list,args.ECE_bin)*100
    
    logger.info('model_A acc: %f'%(test_acc_A))
    logger.info('model_A ECE: %f, AdaECE:%f, ClasswiseECE:%f, Full-ECE:%f'%(ECE_out_A,AdaECE_out_A,Classwise_ECE_out_A,Full_ECE_out_A))
    
    tb_summary_writer.add_scalar('test_loss', test_loss,epoch)
    tb_summary_writer.add_scalar('test_acc_A', test_acc_A,epoch)
    tb_summary_writer.add_scalar('ECE_model_A', ECE_out_A,epoch)
    tb_summary_writer.add_scalar('AdaECE_model_A', AdaECE_out_A,epoch)
    tb_summary_writer.add_scalar('ClasswiseECE_model_A', Classwise_ECE_out_A,epoch)
    tb_summary_writer.add_scalar('Full_ECE_model_A', Full_ECE_out_A,epoch)


    if args.align_flag==True:
        for j, prob_B in enumerate(prob_B_list):
            if args.fudge:
                test_acc_B=test_count_B_list[j]/mask_count*100
            else:
                test_acc_B=test_count_B_list[j]/len(testset)*100
            ECE_out_B=ECE(prob_B_list_list[j],label_list,args.ECE_bin)*100
            AdaECE_out_B=AdaECE(prob_B_list_list[j],label_list,args.ECE_bin)*100
            Classwise_ECE_out_B=Classwise_ECE(prob_B_list_list[j],label_list,args.ECE_bin)*100
            Full_ECE_out_B=Full_ECE(prob_B_list_list[j],label_list,args.ECE_bin)*100
            kl_B_out=test_kl_B/len(testset)
                
            logger.info('model_B%d acc: %f, kl: %f'%(j,test_acc_B,kl_B_out))
            logger.info('model_B%d ECE: %f, AdaECE:%f, ClasswiseECE:%f, Full_ECE:%f'%(j,ECE_out_B,AdaECE_out_B,Classwise_ECE_out_B,Full_ECE_out_B))

            tb_summary_writer.add_scalar('test_kl_B', kl_B_out ,epoch)
            tb_summary_writer.add_scalar('test_acc_B', test_acc_B,epoch)
            tb_summary_writer.add_scalar('ECE_model_B', ECE_out_B,epoch)
            tb_summary_writer.add_scalar('AdaECE_model_B', AdaECE_out_B,epoch)
            tb_summary_writer.add_scalar('ClasswiseECE_model_B', Classwise_ECE_out_B,epoch)
            tb_summary_writer.add_scalar('Full_ECE_model_B', Full_ECE_out_B,epoch)

        tb_summary_writer.add_scalar('align_loss', loss_align ,epoch)
    
    if args.use_conf_model:
        tb_summary_writer.add_scalar('test_loss_conf', test_loss_conf,epoch)
        tb_summary_writer.add_scalar('test_conf_L1', test_conf_L1,epoch)
        logger.info('model_conf L1:%f'%(test_conf_L1))

    return test_acc_A, ECE_out_A



start_epoch=infos['epoch']
test_acc, test_ECE=eval_one_epoch(-1)
for epoch in range(start_epoch,args.epoch):
    train_one_epoch(epoch)
    test_acc, test_ECE=eval_one_epoch(epoch)

    with open(os.path.join(args.checkpoint_path,'model_A.final.th'), 'wb') as f:
        state_dict = model_A.state_dict()
        torch.save(state_dict, f)
    if args.align_flag:
        for j,model_B in enumerate(model_B_list):
            with open(os.path.join(args.checkpoint_path,'model_B%d.final.th'%(j)), 'wb') as f:
                state_dict = model_B.state_dict()
                torch.save(state_dict, f)
    if args.use_conf_model:
        with open(os.path.join(args.checkpoint_path,'model_conf.final.th'), 'wb') as f:
            state_dict = model_conf.state_dict()
            torch.save(state_dict, f)
    
    # if args.count_sample_acc:    
    #     pickle_dump_file(train_sample_count,os.path.join(args.checkpoint_path,'train_count.final.th'))

    pickle_dump_file(infos,os.path.join(args.checkpoint_path,'infos.final.th'))

    if args.baseline_acc>0:
        if infos['stage2']==0:
            if test_acc>infos['overall_acc']:

                infos['overall_acc']=test_acc
                infos['overall_acc-ECE']=test_acc-test_ECE

                with open(os.path.join(args.checkpoint_path,'model_A.overall.th'), 'wb') as f:
                    state_dict = model_A.state_dict()
                    torch.save(state_dict, f)
                if args.align_flag:
                    for j,model_B in enumerate(model_B_list):
                        with open(os.path.join(args.checkpoint_path,'model_B%d.overall.th'%(j)), 'wb') as f:
                            state_dict = model_B.state_dict()
                            torch.save(state_dict, f)
                if args.use_conf_model:
                    with open(os.path.join(args.checkpoint_path,'model_conf.overall.th'), 'wb') as f:
                        state_dict = model_conf.state_dict()
                        torch.save(state_dict, f)
                pickle_dump_file(infos,os.path.join(args.checkpoint_path,'infos.overall.th'))
                logger.info("best overall model is saved")

                if test_acc>args.baseline_acc-args.acc_threshold:
                    infos['stage2']=1
        else:
            if test_acc>args.baseline_acc-args.acc_threshold and test_acc-test_ECE>infos['overall_acc-ECE']:

                infos['overall_acc']=test_acc
                infos['overall_acc-ECE']=test_acc-test_ECE

                with open(os.path.join(args.checkpoint_path,'model_A.overall.th'), 'wb') as f:
                    state_dict = model_A.state_dict()
                    torch.save(state_dict, f)
                if args.align_flag:
                    for j,model_B in enumerate(model_B_list):
                        with open(os.path.join(args.checkpoint_path,'model_B%d.overall.th'%(j)), 'wb') as f:
                            state_dict = model_B.state_dict()
                            torch.save(state_dict, f)
                if args.use_conf_model:
                    with open(os.path.join(args.checkpoint_path,'model_conf.overall.th'), 'wb') as f:
                        state_dict = model_conf.state_dict()
                        torch.save(state_dict, f)
                pickle_dump_file(infos,os.path.join(args.checkpoint_path,'infos.overall.th'))
                logger.info("best overall model is saved")
    
    if test_acc>infos['best_acc']:
        infos['best_acc']=test_acc
        with open(os.path.join(args.checkpoint_path,'model_A.bestacc.th'), 'wb') as f:
            state_dict = model_A.state_dict()
            torch.save(state_dict, f)
        if args.align_flag:
            for j,model_B in enumerate(model_B_list):
                with open(os.path.join(args.checkpoint_path,'model_B%d.bestacc.th'%(j)), 'wb') as f:
                    state_dict = model_B.state_dict()
                    torch.save(state_dict, f)
        if args.use_conf_model:
            with open(os.path.join(args.checkpoint_path,'model_conf.bestacc.th'), 'wb') as f:
                state_dict = model_conf.state_dict()
                torch.save(state_dict, f)
        pickle_dump_file(infos,os.path.join(args.checkpoint_path,'infos.bestacc.th'))
        logger.info("best acc model is saved")

    logger.info("epoch %d finished"%(epoch))

    infos['epoch']+=1

logger.success("Done!")