from .cifar import cifar10, cifar100
from .seqdata import IMDB, AGNews, SST2
from .tiny_imagenet import Tiny_ImageNet
from .imagenet import ImageNet
from .pacs import PACS
from .cifar_c import cifar10_C,cifar100_C
def get_dataset(**kwargs):
    dataset_name=kwargs['dataset']
    if dataset_name=="cifar10":
        dataset=cifar10(**kwargs)
    elif dataset_name=="cifar100":
        dataset=cifar100(**kwargs)
    elif dataset_name=="IMDB":
        dataset=IMDB(**kwargs)
    elif dataset_name=="AGNews":
        dataset=AGNews(**kwargs)
    elif dataset_name=="SST2":
        dataset=SST2(**kwargs)
    elif dataset_name=="Tiny_ImageNet":
        dataset=Tiny_ImageNet(**kwargs)
    elif dataset_name=="ImageNet":
        dataset=ImageNet(**kwargs)
    elif dataset_name=="PACS":
        dataset=PACS(**kwargs)
    elif dataset_name=='cifar10-C':
        dataset=cifar10_C(**kwargs)
    elif dataset_name=='cifar100-C':
        dataset=cifar100_C(**kwargs)
    else:    
        raise ValueError("Unrecognizable dataset type")
    return dataset
