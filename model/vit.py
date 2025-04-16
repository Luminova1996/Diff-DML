import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from loguru import logger
import copy

def random_init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.MultiheadAttention):
        nn.init.xavier_uniform_(m.in_proj_weight)  # 对输入投影矩阵进行 Xavier 初始化
        if m.in_proj_bias is not None:
            nn.init.constant_(m.in_proj_bias, 0)  # 将输入投影的偏置初始化为0
        nn.init.xavier_uniform_(m.out_proj.weight)  # 对输出投影矩阵进行 Xavier 初始化
        if m.out_proj.bias is not None:
            nn.init.constant_(m.out_proj.bias, 0)  # 将输出投影的偏置初始化为0
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, init_zero=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if init_zero:
            self._initialize_weights()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def _initialize_weights(self):
        nn.init.constant_(self.conv1.weight, 0)
        nn.init.constant_(self.conv2.weight, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def get_vit(model_name,**kwargs):
    if model_name=='vit_b_16':
        model=torchvision.models.vit_b_16(weights=None)
    if model_name=='vit_b_32':
        model=torchvision.models.vit_b_32(weights=None)
    if model_name=='vit_l_16':
        model=torchvision.models.vit_l_16(weights=None)
    if model_name=='vit_l_32':
        model=torchvision.models.vit_l_32(weights=None)
    if model_name=='vit_h_14':
        model=torchvision.models.vit_h_14(weights='IMAGENET1K_V1')
    return model

def get_pretrained_vit(model_name,**kwargs):
    if model_name=='vit_b_16':
        model=torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        model.layer_final_in_channels=1024
        model.layer_final_out_channels=1024
    if model_name=='vit_b_32':
        model=torchvision.models.vit_b_32(weights='IMAGENET1K_V1')
        model.layer_final_in_channels=2208
        model.layer_final_out_channels=2208
    if model_name=='vit_l_16':
        model=torchvision.models.vit_l_16(weights='IMAGENET1K_V1')
        model.layer_final_in_channels=1920
        model.layer_final_out_channels=1920
    if model_name=='vit_l_32':
        model=torchvision.models.vit_l_32(weights='IMAGENET1K_V1')
        model.layer_final_in_channels=1920
        model.layer_final_out_channels=1920
    if model_name=='vit_h_14':
        model=torchvision.models.vit_h_14(weights='IMAGENET1K_V1')
        model.layer_final_in_channels=1920
        model.layer_final_out_channels=1920
    model.forward_align=forward_align_vit
    return model

def forward_align_vit(self, input: torch.Tensor):
    # Reshape and permute the input tensor
    x = self._process_input(input)
    n = x.shape[0]

    # Expand the class token to the full batch
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    #x = self.encoder(x)
    x = x + self.encoder.pos_embedding
    for i in range(len(self.encoder.layers)-1):
        x=self.encoder.layers[i](self.encoder.dropout(x))
    x_=self.encoder.layers[-1](self.encoder.dropout(x))
    x_=self.encoder.ln(x_)

    # Classifier "token" as used by standard language architectures
    x_ = x_[:, 0]

    x_ = self.heads(x_)

    return x,x_


class Model_YMTE_same_vit(nn.Module):
    def __init__(self, **kwargs):
        super(Model_YMTE_same_vit, self).__init__()
        self.random_init=kwargs['random_init']
    

    def load_params(self,model):
        self.layer1=copy.deepcopy(model.encoder.layers[-1])
        self.ln=copy.deepcopy(model.encoder.ln)
        self.heads=copy.deepcopy(model.heads)
        if self.random_init:
            self.layer1.apply(random_init_weights)
            self.heads.apply(random_init_weights)

    def forward(self,feat):
        out=feat.detach()
        out=self.layer1(out)
        out=self.ln(out)
        out = out[:, 0]
        out = self.heads(out)
        return out

class Model_YMTE_identity_vit(nn.Module):
    def __init__(self, **kwargs):
        super(Model_YMTE_identity_vit, self).__init__()

    def load_params(self,model):
        self.layer1=copy.deepcopy(model.encoder.layers[-1])
        self.in_planes=model.layer_final_out_channels
        self.layer2=self._make_layer(BasicBlock, self.in_planes, 1, stride=1,init_zero=True)
        self.ln=copy.deepcopy(model.encoder.ln)
        self.heads=copy.deepcopy(model.heads)

    def forward(self,feat):
        out=feat.detach()
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.ln(out)
        out = out[:, 0]
        out = self.heads(out)
        return out

    def _make_layer(self, block, planes, num_blocks, stride,init_zero):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,init_zero))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

class Model_YMTE_zero_vit(nn.Module):
    def __init__(self, **kwargs):
        super(Model_YMTE_zero_vit, self).__init__()
        self.rank=kwargs['lora_rank']

    def load_params(self,model):
        self.in_channels=model.layer_final_in_channels
        self.out_channels=model.layer_final_out_channels
        self.layer1=copy.deepcopy(model.encoder.layers[-1])
        self.layer_final_lora_A=nn.Conv2d(self.in_channels, self.rank, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer_final_lora_B=nn.Conv2d(self.rank, self.out_channels, kernel_size=1, bias=False)
        nn.init.constant_(self.layer_final_lora_B.weight, 0)
        self.ln=copy.deepcopy(model.encoder.ln)
        self.heads=copy.deepcopy(model.heads)

    def forward(self,feat):
        out=feat.detach()
        out_app=self.layer_final_lora_A(out)
        out_app=self.layer_final_lora_B(out_app)        
        out=self.layer1(out)
        out+=out_app
        out=self.ln(out)
        out = out[:, 0]
        out = self.heads(out)
        return out
