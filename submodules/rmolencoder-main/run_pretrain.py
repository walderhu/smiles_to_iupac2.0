import os, random
import pickle

import numpy as np
import torch
import lightning as L

import chython
from chython import smiles

from sklearn.model_selection import train_test_split

import RMolEncoder as rme



### DATA

def split_data_train_test(data, test_size=0.05, random_state=10):
    # a_train, a_test, b_train, b_test, ...
    sd = train_test_split(*data.values(), test_size=test_size, random_state=random_state)
    train_data = {k: v for k, v in zip(data.keys(), sd[::2])}
    test_data = {k: v for k, v in zip(data.keys(), sd[1::2])}

    return train_data, test_data

def make_dataset(data_dict, packType, rename=None, every_n=1):
    Y = {}
    for k, v in data_dict.items():
        if k=="packs":
            continue

        dsk = k
        if rename is not None and k in rename:
            dsk = rename[k]
        
        Y[dsk] = torch.tensor(data_dict[k][::every_n], dtype=torch.float32)
        
    return rme.RxnMolDataset(data_dict["packs"][::every_n], Y, ContainerClass=packType)


FEATURES = {
            "mol_descs":16,
            "rxn_descs":128
           }

# RXN
every_n = 1
print("--- Open RXN data")
with open("./data/ORD/ord_uspto_data.pack.pkl", 'rb') as f:
    rxn_data = pickle.load(f)
rxn_train_data, rxn_valid_data = split_data_train_test(rxn_data, 0.05)

rxn_train_ds = make_dataset(rxn_train_data, chython.ReactionContainer, rename={"desc":"rxn_descs"}, every_n=every_n)
rxn_valid_ds = make_dataset(rxn_valid_data, chython.ReactionContainer, rename={"desc":"rxn_descs"}, every_n=every_n)
print("RXN Datasets:", len(rxn_train_ds), len(rxn_valid_ds))

# MOL
print("--- Open MOL data")
with open("./data/pubchem_data/5KK_pca16_data.pack.pkl", 'rb') as f:
    mol_data = pickle.load(f)
mol_train_data, mol_valid_data = split_data_train_test(mol_data, 0.05)

mol_train_ds = make_dataset(mol_train_data, chython.MoleculeContainer, rename={"desc":"mol_descs"})
mol_valid_ds = make_dataset(mol_valid_data, chython.MoleculeContainer, rename={"desc":"mol_descs"})
print("Mol Datasets:", len(mol_train_ds), len(mol_valid_ds))

## Combine
train_dataset = rme.CombinedDataset(
    rme.PadDataset(rxn_train_ds, FEATURES),
    rme.PadDataset(mol_train_ds, FEATURES)
)

valid_dataset = rme.CombinedDataset(
    rme.PadDataset(rxn_valid_ds, FEATURES),
    rme.PadDataset(mol_valid_ds, FEATURES)
)

print("Combined Datasets:", len(train_dataset), len(valid_dataset))

batch_size = 64
grad_acum = 4

train_loader = rme.make_dataloader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=6)
valid_loader = rme.make_dataloader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=6)

### LOOP

hparams = (
# """
#     dim,  nlayers,  shared, nheads
# """
    # (128,  6,        True,   4),
    # (512,  8,        True,   8),
    # (768,  8,        True,   12),
    (1024,  8,        True,   8),
          )
for d_model, num_in_layers, shared_weights, n_in_head in hparams:
    print(f"--- Model params:\n\tDim: {d_model}\n\tNum layers: {num_in_layers}\n\tShared: {shared_weights}\n\tNum heads: {n_in_head}")

    ### MODEL
    encoder_hparams: dict = dict(
            d_model=d_model, 
            n_in_head=n_in_head,
            num_in_layers=num_in_layers, 
            shared_weights=shared_weights
    )
    mlp_hparams: dict = dict(
             in_dim=d_model,
             dims=[512, 256], drop=0.1, 
             mlp_activation=torch.nn.GELU(), 
    )
    
    print("--- Make model")
    lr_scheduler = rme.get_lr_scheduler(warmup=500, peak=3e-4, c=1e-4, min_lr=2.5e-5, max_lr=1e-3)
    evaluator = rme.Evaluator()
    
    model = rme.model.PretrainModel(encoder_hparams=encoder_hparams,
                                     mlp_hparams=mlp_hparams, 
                                     out_features=FEATURES,
                                     lr_scheduler=lr_scheduler,
                                     optim_kwargs=dict(lr=1, weight_decay=0.05),
                                     evaluator=evaluator
                                    ).cuda()
    
    ### CALLBACKS
    model_name = f"{d_model}d_{n_in_head}h_{num_in_layers}ls_shared-{shared_weights}"
    
    # logger = L.pytorch.loggers.CSVLogger("./logs/", name=model_name)
    logger = L.pytorch.loggers.TensorBoardLogger("./logs/",
                                                 version=model_name,
                                                 default_hp_metric=False)

    
    ### TRAIN
    
    trainer = L.Trainer(
                        precision = "bf16-mixed",
                        max_steps = 1_500_000,
                        # max_epochs=50,
        
                        val_check_interval = 10_000,
                        # limit_val_batches=100,
        
                        log_every_n_steps = 200, 
                        accumulate_grad_batches = grad_acum,

                        gradient_clip_val = 0.01, 
                        gradient_clip_algorithm = 'value',

                        logger = logger,
                        callbacks = 
                        [
                            L.pytorch.callbacks.ModelCheckpoint(
                                dirpath=f"./models/{model_name}",
                                filename="step={step}_ep={epoch}_valid_loss={valid_loss:.6f}",
                                auto_insert_metric_name=False,
                                every_n_train_steps=25_000,
                                save_top_k=-1
                            ),
                            
                            rme.PretrainValidationCallback(),
                            
                            # L.pytorch.callbacks.EarlyStopping(
                            #     monitor="valid_loss",
                            #     min_delta=0.0,
                            #     patience=3,
                            #     mode='min'
                            # )
                          ]
                       )
    
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)




















