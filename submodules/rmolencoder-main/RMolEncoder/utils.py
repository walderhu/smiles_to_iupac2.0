from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import lightning as L

from .dataset import RxnMolDataset
    
    
def get_lr_scheduler(warmup, peak, c, min_lr, max_lr):
    
    def lr_scheduler(n):

        if n<warmup:
            lr = ((peak)/warmup)*n
        else:
            lr = peak*np.exp(-c*(n-warmup))

        if lr>max_lr: lr=max_lr
        if n>warmup and lr<min_lr: lr=min_lr

        return lr
    return lr_scheduler


class PadDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, features, padv=torch.nan):
        self.dataset = dataset
        self.features = features
        self.padv = padv
        self.target = dataset.target

    def __getitem__(self, item: int):
        x, y = self.dataset[item]
        for k, v in self.features.items():
            if k in y: continue
            y[k] = torch.tensor(self.padv) if v==1 else torch.full((v,), self.padv)
                    
        return x, y

    def __len__(self):
        return self.dataset.__len__()


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.length = sum([len(d) for d in datasets])
        self.offsets = np.cumsum([len(d) for d in datasets])
        self.target = True
            
    def __getitem__(self, item: int):
        item = len(self)+item if item<0 else item
        for i, offset in enumerate(self.offsets):
            if item < offset:
                if i > 0:
                    item -= self.offsets[i-1]

                return self.datasets[i][item]
        raise IndexError(f'{item} exceeds {self.length}')

    def __len__(self):
        return self.length


def MSE(pred, y):
    return ((pred - y)**2).mean()

def SSE(pred, y):
    return ((pred - y)**2).sum()

def R2(pred, y):
    yvar = ((y-y.mean())**2).sum()
    pvar = ((pred-y)**2).sum()
    return 1 - pvar/yvar


class Evaluator:
    def __init__(self, weights=None):
        self.weights = weights if weights is not None else {}

    def __call__(self, preds, ys):
        res = {}
        total_loss = 0.
        total_r2 = 0.
        div = 0
        
        for k, y in ys.items():
            p = preds[k]
            take = ~torch.isnan(y)
            if take.sum()==0:
                continue

            p = p[take]
            y = y[take]
            featuren = len(y)
            div += featuren
            
            loss = MSE(p, y)
            res[k+"_loss"] = loss
            total_loss += loss*featuren*self.weights.get(k, 1.)
            
            with torch.no_grad():
                r2 = R2(p, y)
                res[k+"_r2"] = r2
                total_r2 += r2

        res["loss"] = total_loss / div
        res["r2"] = total_r2 / len(ys)
        
        return res


class PretrainValidationCallback(L.Callback):
        
    def on_validation_epoch_end(self, trainer, pl_module):
        preds = {k:torch.cat(v, dim=0) for k, v in pl_module.valid_preds.items()}
        ys =    {k:torch.cat(v, dim=0) for k, v in pl_module.valid_ys.items()}
        
        pl_module.valid_preds.clear()
        pl_module.valid_ys.clear()

        e = pl_module.evaluator(preds, ys)
        for k in e.keys():
            pl_module.log("valid_"+k, float(e[k]))




















