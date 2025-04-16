'''
Pytorch implementation of ResNet models.

Reference:
[1] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR, 2016.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
import torchvision
import copy

NUM_LAYER3_BLOCKS=0

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, init_zero=False):
        super(Bottleneck, self).__init__()
        self.init_zero=init_zero
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
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
        nn.init.constant_(self.conv3.weight, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        #logger.info(torch.sum(torch.abs(out)))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

def random_init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, temp=1.0,**kwargs):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.block_class=block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.temp = temp

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out) / self.temp
        return out

    def forward_align(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        for i in range(len(self.layer3)-NUM_LAYER3_BLOCKS):
            out=self.layer3[i](out)
        out_=out
        for i in range(NUM_LAYER3_BLOCKS):
            out_=self.layer3[i+len(self.layer3)-NUM_LAYER3_BLOCKS](out_)

        #out_ = self.layer3(out)
        out_ = self.layer4(out_)
        out_ = F.adaptive_avg_pool2d(out_, 1)
        out_ = out_.view(out_.size(0), -1)
        out_ = self.fc(out_) / self.temp
        return out,out_

    # def forward_align(self, x):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out_ = self.layer3(out)
    #     out_ = self.layer4(out_)
    #     out_ = F.adaptive_avg_pool2d(out_, 1)
    #     out_ = out_.view(out_.size(0), -1)
    #     out_ = self.fc(out_) / self.temp
    #     return out,out_

class Model_YMTE_same_resnet(nn.Module):
    def __init__(self, **kwargs):
        super(Model_YMTE_same_resnet, self).__init__()
        self.in_planes=512
        self.random_init=kwargs['random_init']
    

    def load_params(self,model):
        if NUM_LAYER3_BLOCKS>0:
            self.layer3_blocks=copy.deepcopy(model.layer3[-1*NUM_LAYER3_BLOCKS:])
        else:
            self.layer3_blocks=nn.Sequential()
        self.layer4=copy.deepcopy(model.layer4)
        self.fc=copy.deepcopy(model.fc)
        
        if self.random_init:
            self.layer3.apply(random_init_weights)
            self.layer4.apply(random_init_weights)
            self.fc.apply(random_init_weights)

    def forward(self,feat):
        out=feat.detach()
        out=self.layer3_blocks(out)
        out=self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class Model_YMTE_identity_resnet(nn.Module):
    def __init__(self, **kwargs):
        super(Model_YMTE_identity_resnet, self).__init__()

    def load_params(self,model):
        if NUM_LAYER3_BLOCKS>0:
            self.layer3_blocks=copy.deepcopy(model.layer3[-1*NUM_LAYER3_BLOCKS:])
        else:
            self.layer3_blocks=nn.Sequential()
        self.layer4=copy.deepcopy(model.layer4)
        if model.block_class==BasicBlock:
            self.in_planes=512
        elif model.block_class==Bottleneck:
            self.in_planes=2048
        self.layer5=self._make_layer(BasicBlock, self.in_planes, 1, stride=1,init_zero=True)
        self.fc=copy.deepcopy(model.fc)

    def forward(self,feat):
        out=feat.detach()
        out=self.layer3_blocks(out)
        out=self.layer4(out)
        out=self.layer5(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _make_layer(self, block, planes, num_blocks, stride,init_zero):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,init_zero))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

class Model_YMTE_zero_resnet(nn.Module):
    def __init__(self, **kwargs):
        super(Model_YMTE_zero_resnet, self).__init__()
        self.in_planes=512
        self.rank=kwargs['lora_rank']

    def load_params(self,model):
        if NUM_LAYER3_BLOCKS>0:
            self.layer3_blocks=copy.deepcopy(model.layer3[-1*NUM_LAYER3_BLOCKS:])
        else:
            self.layer3_blocks=nn.Sequential()
        self.layer4=copy.deepcopy(model.layer4)
        if model.block_class==BasicBlock:
            # self.layer3_lora_A=nn.Conv2d(128, self.rank, kernel_size=3, stride=2, padding=1, bias=False)
            # self.layer3_lora_B=nn.Conv2d(self.rank, 256, kernel_size=1, bias=False)
            # nn.init.constant_(self.layer3_lora_B.weight, 0)
            self.layer4_lora_A=nn.Conv2d(256, self.rank, kernel_size=3, stride=2, padding=1, bias=False)
            self.layer4_lora_B=nn.Conv2d(self.rank,512, kernel_size=1, bias=False)
            nn.init.constant_(self.layer4_lora_B.weight, 0)
        elif model.block_class==Bottleneck:
            # self.layer3_lora_A=nn.Conv2d(128*4, self.rank, kernel_size=3, stride=2, padding=1, bias=False)
            # self.layer3_lora_B=nn.Conv2d(self.rank,256*4, kernel_size=1, bias=False)
            # nn.init.constant_(self.layer3_lora_B.weight, 0)
            self.layer4_lora_A=nn.Conv2d(256*4, self.rank, kernel_size=3, stride=2, padding=1, bias=False)
            self.layer4_lora_B=nn.Conv2d(self.rank,512*4, kernel_size=1, bias=False)
            nn.init.constant_(self.layer4_lora_B.weight, 0)
        self.fc=copy.deepcopy(model.fc)

    def forward(self,feat):
        out=feat.detach()
        # out_app=self.layer3_lora_A(out)
        # out_app=self.layer3_lora_B(out_app)
        #logger.info(out_app.size())
        out=self.layer3_blocks(out)
        #logger.info(out_app.size())
        # out+=out_app
        out_app=self.layer4_lora_A(out)
        out_app=self.layer4_lora_B(out_app)        
        out=self.layer4(out)
        out=out+out_app
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet18(temp=1.0, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], temp=temp, **kwargs)
    return model


def resnet34(temp=1.0, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], temp=temp, **kwargs)
    return model


def resnet50(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], temp=temp, **kwargs)
    return model


def resnet101(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], temp=temp, **kwargs)
    return model


def resnet110(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 26, 3], temp=temp, **kwargs)
    return model


def resnet152(temp=1.0, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], temp=temp, **kwargs)
    return model


def get_pretrained_resnet(model_name,**kwargs):
    if model_name=='resnet50':
        model=torchvision.models.resnet50(weights='IMAGENET1K_V2')
    if model_name=='resnet101':
        model=torchvision.models.resnet101(weights='IMAGENET1K_V2')
    if model_name=='resnet152':
        model=torchvision.models.resnet152(weights='IMAGENET1K_V2')
    model.forward_align=forward_align_resnet
    model.block_class=Bottleneck
    return model

def forward_align_resnet(self,x):
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    
    for i in range(len(self.layer3)-NUM_LAYER3_BLOCKS):
        x=self.layer3[i](x)
    x_=x
    for i in range(NUM_LAYER3_BLOCKS):
        x_=self.layer3[i+len(self.layer3)-NUM_LAYER3_BLOCKS](x_)

    x_ = self.layer4(x_)

    x_ = self.avgpool(x_)
    x_ = torch.flatten(x_, 1)
    x_ = self.fc(x_)

    return x,x_
