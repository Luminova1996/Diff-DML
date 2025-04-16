import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class model_MLP_classifier(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_hidden_layers=0, num_classes=2,**kwargs):
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_hidden_layers=num_hidden_layers
        self.num_classes=num_classes
        self.build_model()
    
    def build_model(self):
        self.Dropout=nn.Dropout(0.2)

        self.embed_net=nn.Linear(in_features=self.input_size,out_features=self.hidden_size)
        
        self.hidden_layers=nn.Sequential()
        for i in range(self.num_hidden_layers):
            self.hidden_layers.add_module('layer_{}'.format(i),nn.Linear(self.hidden_size,self.hidden_size))
            self.hidden_layers.add_module('ReLU_{}'.format(i),nn.ReLU())

        self.score_net=nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.num_classes,bias=False),
        )
    
    def forward(self, input): # h: [batch_size,seq_len,h_dim]
        x=input
        x=x.view(x.size(0),-1)
        x=self.embed_net(x)
        x=self.Dropout(x)
        x=self.hidden_layers(x)
        output=self.score_net(x)
        return output   
    
class model_CNN_classifier(nn.Module):
    def __init__(self, input_size, hidden_size=256, kernel_size=3, hidden_channels=10, num_hidden_layers=0, num_classes=2,**kwargs):
        super().__init__()
        self.input_size=input_size
        self.fc_start_size=input_size*hidden_channels//(3*16)
        self.hidden_size=hidden_size
        self.kernel_size=kernel_size
        self.padding=(kernel_size-1)//2
        self.hidden_channels=hidden_channels
        self.num_hidden_layers=num_hidden_layers
        self.num_classes=num_classes
        self.build_model()
    
    def build_model(self):
        self.Dropout=nn.Dropout(0.2)
        self.pool=nn.MaxPool2d(2)
        self.embed_net=nn.Conv2d(3,self.hidden_channels,kernel_size=self.kernel_size,padding=self.padding)
        
        self.hidden_layers=nn.Sequential()
        for i in range(self.num_hidden_layers):
            self.hidden_layers.add_module('layer_{}'.format(i),nn.Conv2d(self.hidden_channels,self.hidden_channels,kernel_size=self.kernel_size,padding=self.padding))
            self.hidden_layers.add_module('ReLU_{}'.format(i),nn.ReLU())

        self.score_net=nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.fc_start_size,self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.num_classes,bias=False),
        )
    
    def forward(self, input): # h: [batch_size,channels, seq_len,h_dim]
        x=input
        x=self.embed_net(x)
        x=self.pool(x)
        x=self.Dropout(x)
        x=self.hidden_layers(x)
        x=self.pool(x)
        x=x.view(input.size(0),-1)
        output=self.score_net(x)
        return output 
