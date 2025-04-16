import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from loguru import logger
import copy
import torch.nn.init as init

def random_init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

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

def init_weights(module):
    """
    自定义的权重初始化函数。
    对于线性层和卷积层等应用特定的权重初始化；
    其他类型的模块可以根据需要添加处理逻辑。
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):  # 处理包含权重的层
        init.uniform_(module.weight, -0.1, 0.1)  # 使用均匀分布初始化权重
        if module.bias is not None:
            init.constant_(module.bias, 0)  # 初始化偏置为0
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):  # 如果是BatchNorm层
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)

def get_densenet(model_name,**kwargs):
    if model_name=='densenet121':
        model=torchvision.models.densenet121()
        model.classifier=nn.Linear(1024,kwargs['num_classes'])
    if model_name=='densenet161':
        model=torchvision.models.densenet161()
    if model_name=='densenet201':
        model=torchvision.models.densenet201()
    init_weights(model)
    return model

def get_pretrained_densenet(model_name,**kwargs):
    if model_name=='densenet121':
        model=torchvision.models.densenet121(weights='IMAGENET1K_V1')
        model.layer_final_in_channels=1024
        model.layer_final_out_channels=1024
    if model_name=='densenet161':
        model=torchvision.models.densenet161(weights='IMAGENET1K_V1')
        model.layer_final_in_channels=2208
        model.layer_final_out_channels=2208
    if model_name=='densenet201':
        model=torchvision.models.densenet201(weights='IMAGENET1K_V1')
        model.layer_final_in_channels=1920
        model.layer_final_out_channels=1920
    model.forward_align=forward_align_densenet
    return model

def forward_align_densenet(self,x):
    # See note [TorchScript super()]
    x = self.features[0](x)
    x = self.features[1](x)
    x = self.features[2](x)
    x = self.features[3](x)

    x = self.features[4](x)
    x = self.features[5](x)

    x = self.features[6](x)
    x = self.features[7](x)

    x = self.features[8](x)
    x = self.features[9](x) 

    x_ = self.features[10](x)
    x_ = self.features[11](x_) 


    out = F.relu(x_, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = self.classifier(out)
    return x,out


class Model_YMTE_same_densenet(nn.Module):
    def __init__(self, **kwargs):
        super(Model_YMTE_same_densenet, self).__init__()
        self.random_init=kwargs['random_init']
    

    def load_params(self,model):
        self.layer1=copy.deepcopy(model.features[10])
        self.layer2=copy.deepcopy(model.features[11])
        self.classifier=copy.deepcopy(model.classifier)
        if self.random_init:
            self.layer1.apply(random_init_weights)
            self.layer2.apply(random_init_weights)
            self.classifier.apply(random_init_weights)

    def forward(self,feat):
        out=feat.detach()
        out=self.layer1(out)
        out=self.layer2(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class Model_YMTE_identity_densenet(nn.Module):
    def __init__(self, **kwargs):
        super(Model_YMTE_identity_densenet, self).__init__()

    def load_params(self,model):
        self.layer1=copy.deepcopy(model.features[10])
        self.layer2=copy.deepcopy(model.features[11])
        self.in_planes=model.layer_final_out_channels
        self.layer3=self._make_layer(BasicBlock, self.in_planes, 1, stride=1,init_zero=True)
        self.classifier=copy.deepcopy(model.classifier)

    def forward(self,feat):
        out=feat.detach()
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def _make_layer(self, block, planes, num_blocks, stride,init_zero):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,init_zero))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

class Model_YMTE_zero_densenet(nn.Module):
    def __init__(self, **kwargs):
        super(Model_YMTE_zero_densenet, self).__init__()
        self.rank=kwargs['lora_rank']

    def load_params(self,model):
        self.in_channels=model.layer_final_in_channels
        self.out_channels=model.layer_final_out_channels
        self.layer1=copy.deepcopy(model.features[10])
        self.layer2=copy.deepcopy(model.features[11])
        self.layer_final_lora_A=nn.Conv2d(self.in_channels, self.rank, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer_final_lora_B=nn.Conv2d(self.rank, self.out_channels, kernel_size=1, bias=False)
        nn.init.constant_(self.layer_final_lora_B.weight, 0)
        self.classifier=copy.deepcopy(model.classifier)

    def forward(self,feat):
        out=feat.detach()
        out_app=self.layer_final_lora_A(out)
        out_app=self.layer_final_lora_B(out_app)        
        out=self.layer1(out)
        out+=out_app
        out=self.layer2(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


