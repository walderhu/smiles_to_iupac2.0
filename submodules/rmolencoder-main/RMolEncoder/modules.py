from typing import Iterable, Union
from math import inf

import torch
import torch.nn as nn
from torchtyping import TensorType

import numpy as np
import chytorch

from chytorch.nn.molecule import MoleculeEncoder
from chytorch.utils.data import ReactionEncoderDataBatch
from .dataset import ROLE



class EmbeddingMoleculeEncoder(MoleculeEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings, distances) -> TensorType['batch', 'atoms', 'embedding']:
        """
        Use 0 for padding.
        Atoms should be coded by atomic numbers + 2.
        Token 1 reserved for cls token, 2 reserved for molecule cls or training tricks like MLM.
        Neighbors should be coded from 2 (means no neighbors) to max neighbors + 2.
        Neighbors equal to 1 reserved for training tricks like MLM. Use 0 for cls.
        Distances should be coded from 2 (means self-loop) to max_distance + 2.
        Non-reachable atoms should be coded by 1.
        """
        x = embeddings
        
        for lr, d in zip(self.layers, self.distance_encoders):
            if d is not None:
                d_mask = d(distances).permute(0, 3, 1, 2)  # BxNxNxH > BxHxNxN
            # else: reuse previously calculated mask
            x, _ = lr(x, d_mask)  # noqa

        if self.post_norm:
            return self.norm(x)
        return x


class ChemEncoder(torch.nn.Module):
    def __init__(self, 
                 d_model: int = 256, 
                 max_neighbors: int = 14, 
                 max_distance: int = 10, 
                 n_in_head: int = 8, 
                 num_in_layers: int = 8, 
                 shared_weights: bool = True, 
                 dim_feedforward_ratio: int = 4, 
                 dropout: float = 0.1, 
                 activation=torch.nn.GELU, 
                 layer_norm_eps: float = 1e-5):
        """
        Reaction Graphormer from https://doi.org/10.1021/acs.jcim.2c00344.

        :param max_neighbors: maximum atoms neighbors count.
        :param max_distance: maximal distance between atoms.
        :param num_in_layers: intramolecular layers count
        """
        super().__init__()

        dim_feedforward = dim_feedforward_ratio*d_model
        self.role_encoder = torch.nn.Embedding(len(ROLE), d_model, 0)
        self.molecule_encoder = EmbeddingMoleculeEncoder(max_neighbors=max_neighbors, max_distance=max_distance, d_model=d_model,
                                                        nhead=n_in_head, num_layers=num_in_layers, shared_weights=shared_weights,
                                                        dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
                                                        layer_norm_eps=layer_norm_eps)
    

    def forward(self, batch: ReactionEncoderDataBatch):
        atoms, neighbors, distances, roles = batch

        x = self.molecule_encoder.embedding(atoms, neighbors)
        x = x + self.role_encoder(roles)
        x = self.molecule_encoder(x, distances)
        x = x * (roles > 0).unsqueeze_(-1)
        cls_emb, atom_embs = x[:, 0], x[:, 1:]
        
        return cls_emb, atom_embs
    
    
class MLP(nn.Module):
    def __init__(self, in_dim, dims, drop, mlp_activation):
        super().__init__()
        
        self.mlps = nn.ModuleList()
        for i, d in enumerate(dims):
            in_dim = in_dim if i==0 else dims[i-1] 
            l = nn.Linear(in_dim, d)
            torch.nn.init.kaiming_uniform_(l.weight, nonlinearity='relu')
            
            mlp_block = nn.Sequential(l, mlp_activation, nn.Dropout(drop))
            self.mlps.append(mlp_block)
        
    def forward(self, x):
        for l in self.mlps:
            x = l(x)
        
        return x


class RMolEmbedder(torch.nn.Module):
    def __init__(self,
                 encoder_hparams: dict,
                 mlp_hparams: dict
                ):
        
        super().__init__()
        self.encoder = ChemEncoder(**encoder_hparams)
        self.mlp = MLP(**mlp_hparams)

    
    def forward(self, x):
        cls_emb, atom_embs = self.encoder(x)
        mlp_emb = self.mlp(cls_emb)
        
        return mlp_emb, cls_emb


class Head(nn.Module):
    def __init__(self, in_dim, features):
        super().__init__()

        self.keys = list(features.keys())
        self.layers = nn.ModuleList()
        for n in features.values():
            l = nn.Linear(in_dim, n)
            torch.nn.init.xavier_uniform_(l.weight, gain=1.0)
            self.layers.append(l)

    def forward(self, x):
        return {k: l(x).squeeze(1) for k, l in zip(self.keys, self.layers)}








