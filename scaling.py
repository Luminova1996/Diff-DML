import torch
from torch import nn
import numpy as np
from tqdm import tqdm

class Temperature(nn.Module):
    def __init__(self):
        super(Temperature, self).__init__()
        self.temp = nn.Parameter(data=torch.tensor(1.5))
        
    def forward(self, x):
        return x / self.temp

class Vector(nn.Module,):
    def __init__(self, vec_len, temperature_ref, vec_reg=0.1):
        super(Vector, self).__init__()
        self.vec = nn.Parameter(data=torch.ones(vec_len)/temperature_ref)
        self.vec_reg = vec_reg
    
    def forward(self, x):
        return x * self.vec

class Dirichlet(nn.Module):
    def __init__(self, num_classes, temperature_ref, off_diag_reg=1, bias_reg=1, diag_reg=0.1):
        super(Dirichlet, self).__init__()
        self.model = nn.Sequential(nn.Linear(num_classes, num_classes))
        self.model[0].weight = torch.nn.Parameter(torch.eye(num_classes)/temperature_ref)
        self.model[0].bias = torch.nn.Parameter(torch.zeros((num_classes,)))
        self.off_diag_reg = off_diag_reg
        self.bias_reg = bias_reg
        self.diag_reg = diag_reg
        
    def forward(self, x):
        return self.model(x)

def iter_printer(iterater):
    return tqdm(iterater,mininterval=5,maxinterval=100,miniters=5)

def fit_scaling_model(method, dataloader_logits_calib, num_classes, binary_loss, regularization, temperature_ref=1, num_epochs=200):
    
    if method == 'temperature':
        calibrator = Temperature().cuda()
    elif method =='vector':
        calibrator = Vector(num_classes, temperature_ref).cuda()
    elif method == 'dirichlet':
        calibrator = Dirichlet(num_classes, temperature_ref).cuda()
    else:
        raise ValueError('Unknown method')

    optimizer = torch.optim.Adam(calibrator.parameters(), lr=0.001)

    for epoch in iter_printer(range(num_epochs)):
        epoch_loss = 0
        for x, y in dataloader_logits_calib:
            optimizer.zero_grad()
            x, y = x.cuda(), y.cuda()
            logits_scaled = calibrator(x)
            if binary_loss:
                probas = torch.softmax(logits_scaled, axis=1)
                confidence, y_pred = torch.max(probas, axis=1)
                correct = (y_pred == y).float()
                loss = nn.functional.binary_cross_entropy(confidence, correct)
            else:
                loss = nn.functional.cross_entropy(logits_scaled, y)
            if method == 'dirichlet':
                loss += calibrator.off_diag_reg * calibrator.model[0].weight.clone().fill_diagonal_(0).square().sum()
                loss += calibrator.bias_reg * calibrator.model[0].bias.square().sum()
                if regularization:
                    loss += calibrator.diag_reg * (torch.diagonal(calibrator.model[0].weight) - 1).square().mean()
            elif method == 'vector' and regularization:
                loss += calibrator.vec_reg * (calibrator.vec - 1).square().mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
    
    return calibrator