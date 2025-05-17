from typing import Iterable, Union
from collections import defaultdict

import torch
import torch.nn as nn

import numpy as np
import lightning as L

from .utils import get_lr_scheduler, PadDataset, CombinedDataset, Evaluator
from .modules import ChemEncoder, MLP, RMolEmbedder, Head



class PretrainModel(L.LightningModule):
    def __init__(self,
                 encoder_hparams: dict,
                 mlp_hparams: dict,
                 out_features: dict,
                 
                 lr_scheduler=get_lr_scheduler(warmup=500, peak=5e-4, c=1e-4, min_lr=5e-5, max_lr=1e-3),
                 optim=torch.optim.AdamW, 
                 optim_kwargs=dict(lr=1, weight_decay=0.05),
                 evaluator=None
                ):
        
        super().__init__()
        self.save_hyperparameters()
        
        self.lr_scheduler = lr_scheduler
        self.optim = optim
        self.optim_kwargs = optim_kwargs
        self.evaluator = evaluator
        
        self.valid_preds = defaultdict(list)
        self.valid_ys = defaultdict(list)
        
        self.embedder = RMolEmbedder(encoder_hparams, mlp_hparams)
        self.head = Head(mlp_hparams["dims"][-1], out_features)
        
        
    def forward(self, x, with_emb=False):
        mlp_emb, cls_emb = self.embedder(x)
        x = self.head(mlp_emb)
        
        if with_emb:
            x.update({"emb":cls_emb})
        return x

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        e = self.evaluator(pred, y)

        self.log_dict(
            {k:float(v) for k, v in e.items()},
            on_step=True,
            logger=True,
            sync_dist=True,
        )
        
        return e['loss']
        

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            pred = self(x)

        for k in pred.keys():
            self.valid_preds[k].append(pred[k].cpu())
            self.valid_ys[k].append(y[k].cpu())
           
    
    def configure_optimizers(self):
        optimizer = self.optim(self.parameters(), **self.optim_kwargs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_scheduler)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step",
                "frequency": 1,
            },
        }