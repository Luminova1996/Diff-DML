from .model import model_MLP_classifier, model_CNN_classifier
from .resnet import get_pretrained_resnet,resnet18,resnet34,resnet50,resnet101,resnet110,resnet152,Model_YMTE_same_resnet,Model_YMTE_identity_resnet,Model_YMTE_zero_resnet
from .densenet import get_densenet,get_pretrained_densenet,Model_YMTE_same_densenet,Model_YMTE_identity_densenet,Model_YMTE_zero_densenet
from .vit import get_vit,get_pretrained_vit,Model_YMTE_same_vit,Model_YMTE_identity_vit,Model_YMTE_zero_vit
from .seqmodel import model_GRU_classifier,model_LSTM_classifier, model_GPT2_classifier

def get_model(**kwargs):
    model_type=kwargs['model_type']

    if model_type=="MLP":
        model=model_MLP_classifier(**kwargs)
    elif model_type=="CNN":
        model=model_CNN_classifier(**kwargs)
    elif model_type=="resnet18":
        model=resnet18(**kwargs)
    elif model_type=="resnet34":
        model=resnet34(**kwargs)
    elif model_type=="resnet50":
        model=resnet50(**kwargs)
    elif model_type=="resnet101":
        model=resnet101(**kwargs)
    elif model_type=="resnet110":
        model=resnet110(**kwargs)
    elif model_type=="resnet152":
        model=resnet152(**kwargs)
    elif model_type=="resnet50-pretrained":
        model=get_pretrained_resnet('resnet50',**kwargs)
    elif model_type=="resnet101-pretrained":
        model=get_pretrained_resnet('resnet101',**kwargs)
    elif model_type=="resnet152-pretrained":
        model=get_pretrained_resnet('resnet152',**kwargs)
    elif model_type=="densenet121":
        model=get_densenet('densenet121',**kwargs)
    elif model_type=="densenet161":
        model=get_densenet('densenet161',**kwargs)
    elif model_type=="densenet201":
        model=get_densenet('densenet201',**kwargs)
    elif model_type=="densenet121-pretrained":
        model=get_pretrained_densenet('densenet121',**kwargs)
    elif model_type=="densenet161-pretrained":
        model=get_pretrained_densenet('densenet161',**kwargs)
    elif model_type=="densenet201-pretrained":
        model=get_pretrained_densenet('densenet201',**kwargs)
    elif model_type=="vit-b-16":
        model=get_vit('vit_b_16',**kwargs)
    elif model_type=="vit-b-32":
        model=get_vit('vit_b_32',**kwargs)
    elif model_type=="vit-l-16":
        model=get_vit('vit_l_16',**kwargs)
    elif model_type=="vit-l-32":
        model=get_vit('vit_l_32',**kwargs)
    elif model_type=="vit-h-14":
        model=get_vit('vit_h_14',**kwargs)
    elif model_type=="vit-b-16-pretrained":
        model=get_pretrained_vit('vit_b_16',**kwargs)
    elif model_type=="vit-b-32-pretrained":
        model=get_pretrained_vit('vit_b_32',**kwargs)
    elif model_type=="vit-l-16-pretrained":
        model=get_pretrained_vit('vit_l_16',**kwargs)
    elif model_type=="vit-l-32-pretrained":
        model=get_pretrained_vit('vit_l_32',**kwargs)
    elif model_type=="vit-h-14-pretrained":
        model=get_pretrained_vit('vit_h_14',**kwargs)
    elif model_type=="LSTM":
        model=model_LSTM_classifier(**kwargs)
    elif model_type=="GRU":
        model=model_GRU_classifier(**kwargs)
    elif model_type=="GPT2":
        model=model_GPT2_classifier(**kwargs)
    elif model_type=="YMTE-same-resnet":
        model=Model_YMTE_same_resnet(**kwargs)
    elif model_type=="YMTE-identity-resnet":
        model=Model_YMTE_identity_resnet(**kwargs)
    elif model_type=="YMTE-zero-resnet":
        model=Model_YMTE_zero_resnet(**kwargs)
    elif model_type=="YMTE-same-densenet":
        model=Model_YMTE_same_densenet(**kwargs)
    elif model_type=="YMTE-identity-densenet":
        model=Model_YMTE_identity_densenet(**kwargs)
    elif model_type=="YMTE-zero-densenet":
        model=Model_YMTE_zero_densenet(**kwargs)
    elif model_type=="YMTE-same-vit":
        model=Model_YMTE_same_vit(**kwargs)
    elif model_type=="YMTE-identity-vit":
        model=Model_YMTE_identity_vit(**kwargs)
    elif model_type=="YMTE-zero-vit":
        model=Model_YMTE_zero_vit(**kwargs)

    else:    
        raise ValueError("Unrecognizable model type")
    return model
