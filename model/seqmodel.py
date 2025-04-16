import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, AutoConfig, AutoModel
from loguru import logger
import sys
sys.path.append('../')
from utils import *


class model_GRU_classifier(nn.Module):
    def __init__(self, input_size, hidden_size=256,num_classes=2,**kwargs):
        super().__init__()
        self.pred_from_first=True
        self.tokenizer=kwargs['tokenizer']
        self.config=kwargs['config']
        self.vocab_size=self.config.vocab_size
        self.h_dim=self.config.n_embd
        self.w_dim=self.config.n_embd
        self.hidden_size=hidden_size
        self.num_classes=num_classes
        self.fudge=kwargs['fudge']
        self.build_model()
    
    def build_model(self):
        self.Dropout=nn.Dropout(0.2)
        self.embedding=nn.Embedding(self.config.vocab_size,self.w_dim)

        self.h_w_embed_net=nn.GRUCell(input_size=self.w_dim,hidden_size=self.hidden_size)
        
        self.score_net=nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.num_classes,bias=False),
        )
    
    def forward(self, seq,**kwargs): # h: [batch_size,seq_len,h_dim]
        h=self.get_h(seq)
        if not self.fudge:
            mask=(seq!=self.tokenizer.eos_token_id).type_as(seq)
            final_index=torch.sum(mask,dim=1)-1  #[batch_size]
            h=take_index(h.permute(0,2,1),final_index[:,None,None].expand(-1,h.size(-1),1)).squeeze(-1) #[batch_size,h_dim]
        h=self.Dropout(h)
        output=self.score_net(h)
        return output
    
    def get_h(self,seq,**kwargs):
        h=torch.zeros(seq.size(0),self.hidden_size).cuda()
        h_list=[]
        seq_embed=self.embedding(seq)
        for i in range(seq.size(1)):
            h=self.h_w_embed_net(seq_embed[:,i],h)
            h_list.append(h.unsqueeze(1))
        h_list=torch.cat(h_list,dim=1)
        return h_list
        
class model_LSTM_classifier(nn.Module):
    def __init__(self, input_size, hidden_size=256,num_classes=2,**kwargs):
        super().__init__()
        self.pred_from_first=True
        self.tokenizer=kwargs['tokenizer']
        self.config=kwargs['config']
        self.vocab_size=self.config.vocab_size
        self.h_dim=self.config.n_embd
        self.w_dim=self.config.n_embd
        self.hidden_size=hidden_size
        self.num_classes=num_classes
        self.fudge=kwargs['fudge']
        self.build_model()
    
    def build_model(self):
        self.Dropout=nn.Dropout(0.2)
        self.embedding=nn.Embedding(self.config.vocab_size,self.w_dim)

        self.h_w_embed_net=nn.LSTMCell(input_size=self.w_dim,hidden_size=self.hidden_size)
        
        self.score_net=nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.num_classes,bias=False),
        )
    
    def forward(self, seq,**kwargs): # h: [batch_size,seq_len,h_dim]
        h=self.get_h(seq)
        if not self.fudge:
            mask=(seq!=self.tokenizer.eos_token_id).type_as(seq)
            final_index=torch.sum(mask,dim=1)-1  #[batch_size]
            h=take_index(h.permute(0,2,1),final_index[:,None,None].expand(-1,h.size(-1),1)).squeeze(-1) #[batch_size,h_dim]
        h=self.Dropout(h)
        output=self.score_net(h)
        return output
    
    def get_h(self,seq,**kwargs):
        h=torch.zeros(seq.size(0),self.hidden_size).cuda()
        c=torch.zeros(seq.size(0),self.hidden_size).cuda()
        h_list=[]
        seq_embed=self.embedding(seq)
        for i in range(seq.size(1)):
            h,c=self.h_w_embed_net(seq_embed[:,i],(h,c))
            h_list.append(h.unsqueeze(1))
        h_list=torch.cat(h_list,dim=1)
        return h_list


class model_GPT2_classifier(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_hidden_layers=0, num_classes=2,**kwargs):
        super().__init__()
        self.init_model=kwargs['init_model']
        self.tokenizer=kwargs['tokenizer']
        self.config=kwargs['config']
        self.vocab_size=self.config.vocab_size
        self.h_dim=self.config.n_embd
        self.w_dim=self.config.n_embd
        self.pred_from_first=True
        self.hidden_size=hidden_size
        self.num_classes=num_classes
        self.num_hidden_layers=num_hidden_layers
        self.fudge=kwargs['fudge']
        self.build_model()

    
    def build_model(self):
        self.LMmodel=GPT2Model.from_pretrained(self.init_model).cuda()
       
        self.Dropout=nn.Dropout(0.2)

        self.h_embed_net=nn.Sequential(
            nn.Linear(self.h_dim,self.hidden_size),
        )
        
        self.hidden_layers=nn.Sequential()
        for i in range(self.num_hidden_layers):
            self.hidden_layers.add_module('layer_{}'.format(i),nn.Linear(self.hidden_size,self.hidden_size))
            self.hidden_layers.add_module('ReLU_{}'.format(i),nn.ReLU())
            
        self.score_net=nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.num_classes,bias=False),
        )
    
    def forward(self, seq, **kwargs): # seq:[batch_size,seq_len] h: [batch_size,seq_len,h_dim]

        h=self.get_h(seq)
        if not self.fudge:
            mask=(seq!=self.tokenizer.eos_token_id).type_as(seq)
            final_index=torch.sum(mask,dim=1)-1  #[batch_size]
            h=take_index(h.permute(0,2,1),final_index[:,None,None].expand(-1,h.size(-1),1)).squeeze(-1) #[batch_size,h_dim]
        h=self.Dropout(h)
        output=self.h_embed_net(h)
        for layer in self.hidden_layers:
            output=layer(output)
        output=self.score_net(output)

        return output
    
    def get_h(self,seq,**kwargs):
        LM_kwargs={
            'input_ids':seq,
        }
        with torch.no_grad():
            LM_hidden=self.LMmodel(**LM_kwargs)[0]  # [batch_size,seq_len,h_dim]
        return LM_hidden

