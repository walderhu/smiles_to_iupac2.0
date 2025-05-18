from typing import Sequence, Union, Optional
from enum import Enum
import itertools

import torch
from chython import smiles
from chython import ReactionContainer, MoleculeContainer
from chytorch.utils.data.molecule.encoder import MoleculeDataset
from chytorch.utils.data.reaction.encoder import ReactionEncoderDataPoint
from chytorch.utils.data import collate_encoded_reactions

from chython.exceptions import IncorrectSmiles


class ROLE(Enum):
    PAD = 0
    CLS = 1
    REACTANT = 2
    PRODUCT = 3
    NO_RXN = 4


class RxnMolDataset(torch.utils.data.Dataset):
    __slots__ = ("data", "target", "max_distance", "max_neighbors", "cache_points", "ContainerClass")
    
    def __init__(self,
                 data: Sequence[Union[str, ReactionContainer, MoleculeContainer]],
                 target: Union[torch.Tensor, dict[torch.Tensor]] = None,
                 max_distance: int = 10,
                 max_neighbors: int = 14,
                 cache_points: bool = True,
                 ContainerClass: Optional = None
                 ):
        """
        :param data: reactions/molecules collection
        :param target: optional target values
        :param max_distance: set distances greater than cutoff to cutoff value
        :param max_neighbors: set neighbors count greater than cutoff to cutoff value
        """
        # self.data = list(map(lambda x: (smiles(x) if isinstance(x, str) else x), data))
        ##
        self.data = []
        for x in data:
            if isinstance(x, str):
                try:
                    self.data.append(smiles(x))
                except IncorrectSmiles:
                    pass
                except Exception as exc:
                    print(exc)
            else:
                self.data.append(x)
        ##
        self.target = target
        self.max_distance = max_distance
        self.max_neighbors = max_neighbors
        self.cache_points = cache_points
        self.ContainerClass = ContainerClass

    
    def make_data_point(self, x: Union[ReactionContainer, MoleculeContainer, bytes]):
        if isinstance(x, bytes):
            x = self.ContainerClass.unpack(x)
        
        if isinstance(rxn:=x, ReactionContainer):
            mols = rxn.reactants + rxn.products
            molecule_roles = [ROLE.REACTANT.value]*len(rxn.reactants) + [ROLE.PRODUCT.value]*len(rxn.products)
        elif isinstance(mol:=x, MoleculeContainer):
            mols = mol.split()
            molecule_roles = [ROLE.REACTANT.value]*len(mols)
        else:
            raise ValueError(f"Invalid data - {type(x)}")
            
        molecules = MoleculeDataset(mols, max_distance=self.max_distance,
                                    max_neighbors=self.max_neighbors, add_cls=False)

        atoms, neighbors, roles = [torch.IntTensor([1])], [torch.IntTensor([0])], [ROLE.CLS.value]
        distances = []
        for i, (m, r) in enumerate(zip(mols, molecule_roles)):
            a, n, d = molecules[i]
            atoms.append(a)
            neighbors.append(n)
            distances.append(d)
            roles.extend(itertools.repeat(r, len(m)))

        tmp = torch.ones(len(roles), len(roles), dtype=torch.int32)
        offset = 1
        for d in distances:
            j = offset + d.size(0)
            tmp[offset:j, offset:j] = d
            offset = j

        data_point = ReactionEncoderDataPoint(torch.cat(atoms), torch.cat(neighbors), tmp, torch.IntTensor(roles))
        return data_point


    def take_target_point(self, item: int):
        if isinstance(self.target, dict):
            return {k:v[item] for k, v in self.target.items()}

        return self.target[item]
        
        
    def __getitem__(self, item: int) -> ReactionEncoderDataPoint:
        data_point = self.data[item]
        if isinstance(data_point, (ReactionContainer, MoleculeContainer, bytes)):
            data_point = self.make_data_point(data_point)
            if self.cache_points:
                self.data[item] = data_point
            
        if self.target is not None:
            return data_point, self.take_target_point(item)
        return data_point

    def __len__(self):
        return len(self.data)


def make_dataloader(
                    dataset: RxnMolDataset, 
                    batch_size: int,
                    shuffle=False, drop_last=False,
                    **kwargs
                   ):

    def collate_with_target(batch):
        x = [b[0] for b in batch]
        ys = [b[1] for b in batch]

        if isinstance(ys[0], dict):
            Y = {}
            for k, v in ys[0].items():
                fys = [y[k] for y in ys]
                Y[k] = torch.stack(fys)
        else: # Tensor
            Y = torch.stack(ys)
                
        return collate_encoded_reactions(x), Y

    collate_fn = collate_encoded_reactions
    if dataset.target is not None:
        collate_fn = collate_with_target
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                       collate_fn=collate_fn, 
                                       shuffle=shuffle, drop_last=drop_last,
                                       **kwargs
                                      )





