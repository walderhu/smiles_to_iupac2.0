from statistics import mean
import logging
import os
import random
import warnings
from io import StringIO
from os.path import exists, join, basename
from __features import err_wrap
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import RMolEncoder as rme
import torch
import torch.nn as nn
from lithium import send_msg, send_photo
from torch.cuda.amp import GradScaler
from __features import *

from _model import ChemLM, Loss, batch_reader, make_fix_len_collate, tokenize


def train_epoch(model, files, batch_size, epoch, trainer):
    losses = []
    random.shuffle(files)
    for file in files:
        meta = lambda: f"{cpu()}, {ram()}, {gpu()}"
        with tqdm(total=count_lines(file), desc=meta(), dynamic_ncols=True) as tq:
            for num_batch, batch in enumerate(batch_reader(file, batch_size), start=1):
                try:
                    df = pd.read_csv(StringIO(batch), delimiter=';', encoding='utf-8').dropna()
                except Exception as exc:
                    logging.error(exc)
                    continue

                loss = trainer.train(df)
                losses.append(loss)
                logging.info(f"Epoch {epoch}, File: {basename(file)}, Batch {num_batch}, Loss: {loss:.4f}, MeanLoss: {mean(losses):.4f}")
                tq.set_description(meta())
                tq.update(batch_size)
    return losses


def validate_and_save(model, trainer, validation_data, epoch, best_val_loss, \
                            patience, epochs_without_improvement):
    val_loss = trainer.validate(validation_data)
    logging.info(f"Epoch {epoch}, Validation loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "model.pth")
        logging.info(f"Epoch {epoch}: New best validation loss: {best_val_loss:.4f}")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            logging.info(f"Early stopping triggered! No improvement for {patience} epochs.")
    return best_val_loss, epochs_without_improvement


def train(patience=3, batch_size=256, num_epoch=10, pretrained_path = None, dirname = None):  # Added dirname and pretrained_path as parameters
    model = ChemLM(chem_encoder_params=dict(d_model=512, n_in_head=8, num_in_layers=8, shared_weights=True),
                   decoder_params=dict(d_model=512, nhead=8, dim_feedforward=4 * 512, dropout=0.1,
                                       activation=nn.functional.gelu, batch_first=True, norm_first=True, bias=True),
                   num_layers=4, vocab_size=128, chem_encoder_pretrained_path=pretrained_path)
    model.train()
    files = [join(dirname, 'train_Compound_000000001_000500000.csv')] 
    validation_data = pd.read_csv(join('__val__', 'validation_Compound_000000001_000500000.csv'),
                                 delimiter=';', encoding='utf-8').dropna()
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    trainer = Trainer(model) 
    for epoch in range(1, num_epoch):
        losses = train_epoch(model, files, batch_size, epoch, trainer)
        if mean(losses) < 0.1:
            logging.info("BETTER!")
            torch.save(model.state_dict(), "model.pth") 

        best_val_loss, epochs_without_improvement = validate_and_save(model, trainer, validation_data, \
                                            epoch, best_val_loss, patience, epochs_without_improvement)
        if epochs_without_improvement >= patience:
            break
        trainer.send_result(telegram=True, local_save=False)