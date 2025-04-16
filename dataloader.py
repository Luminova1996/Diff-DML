from torch.utils.data import DataLoader

def get_dataloader(dataset, mode='train', **kwargs):
    dataloader_kwargs={
        'dataset':dataset,
        'batch_size':kwargs["batch_size"],
        'num_workers':0,
        'shuffle':True if mode=='train' else False,
    }
    dataloader=DataLoader(**dataloader_kwargs)
    return dataloader
