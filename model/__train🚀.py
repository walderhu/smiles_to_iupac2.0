import logging
import os
import random
import warnings
from io import StringIO
from os.path import exists, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import RMolEncoder as rme
import torch
import torch.nn as nn
from lithium import send_msg, send_photo
from torch.cuda.amp import GradScaler

from _model import ChemLM, Loss, batch_reader, make_fix_len_collate, tokenize

# Мета настройка
send_msg(f'{__file__}: Рассчеты начались', delete_after=5)

dirname = '__train__'
if not exists(dirname):
    raise FileNotFoundError(f"Нет папки с обучающей выборкой по пути: {dirname}")

pretrained_path = "pretrained_encoder.pth"
if not exists(pretrained_path):
    raise FileNotFoundError(f"Предобученный файл не найден по пути: {pretrained_path}")

log_filename = 'training.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')
if exists(log_filename):
    with open(log_filename, 'w'):
        pass

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=FutureWarning)


class Trainer:
    def __init__(self, model: ChemLM):
        self.model = model
        self.scaler = GradScaler()
        self.model.encoder.requires_grad_(False)
        self.model = torch.compile(self.model)
        self.model = self.model.to(device)
        self.losses = []
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)

    def train(self, df: pd.DataFrame):
        MAX_SEQ_LEN = df["IUPAC Name"].apply(len).max()
        DS_SEQ_LEN = MAX_SEQ_LEN + 3
        smis = list(df['SMILES'])
        seqs = [tokenize(df.iloc[i]["IUPAC Name"], add_spezial_tokens=True) for i in range(df.shape[0])]

        ds = rme.dataset.RxnMolDataset(smis, seqs)
        dl = torch.utils.data.DataLoader(ds, batch_size=128,
                                         collate_fn=make_fix_len_collate(DS_SEQ_LEN), shuffle=True, drop_last=False)
        losses = []
        for b in dl:
            x, y = b
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                p = self.model(x, y[:, :-1])
                loss = Loss(p, y[:, 1:])

            losses.append(float(loss))
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.optim.zero_grad()
            self.scaler.update()

        self.losses.extend(losses)
        return np.mean(losses)

    def validate(self, df: pd.DataFrame):
        self.model.eval()
        MAX_SEQ_LEN = df["IUPAC Name"].apply(len).max()
        DS_SEQ_LEN = MAX_SEQ_LEN + 3
        smis = list(df['SMILES'])
        seqs = [tokenize(df.iloc[i]["IUPAC Name"], add_spezial_tokens=True) for i in range(df.shape[0])]
        ds = rme.dataset.RxnMolDataset(smis, seqs)
        dl = torch.utils.data.DataLoader(ds, batch_size=128, collate_fn=make_fix_len_collate(DS_SEQ_LEN),
                                         shuffle=False, drop_last=False)
        losses = []
        with torch.no_grad():
            for b in dl:
                x, y = b
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    p = self.model(x, y[:, :-1])
                    loss = Loss(p, y[:, 1:])
                losses.append(float(loss))
        self.model.train()
        return np.mean(losses)

    def send_result(self, telegram=True, *, local_save=False,
                    filename='training_.png', caption: str = "Training Loss",
                    xlabel='Iteration', ylabel='Loss', ylim: tuple = None):
        if telegram:
            send_photo(self.losses)
        if local_save:
            plt.figure()
            plt.plot(self.losses)
            if ylim:
                plt.ylim(ylim)
            plt.title(caption)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.savefig(filename, format='png')
            plt.close()


def train(patience=10):
    model = ChemLM(chem_encoder_params=dict(d_model=512, n_in_head=8, num_in_layers=8, shared_weights=True),
                   decoder_params=dict(d_model=512, nhead=8, dim_feedforward=4 * 512, dropout=0.1,
                                       activation=nn.functional.gelu, batch_first=True, norm_first=True, bias=True),
                   num_layers=4, vocab_size=128, chem_encoder_pretrained_path=pretrained_path)
    model.train()
    files = [join(dirname, 'train_Compound_000000001_000500000.csv')]
    validation_data = pd.read_csv(join('__val__', 'validation_Compound_000000001_000500000.csv'), delimiter=';', encoding='utf-8').dropna()
    loss = float('inf')
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(1, 100):
        random.shuffle(files)
        trainer = Trainer(model)
        for file in files:
            for num_batch, batch in enumerate(batch_reader(file, batch_size=256), start=1):  # 1e4
                try:
                    df = pd.read_csv(StringIO(batch), delimiter=';', encoding='utf-8').dropna()
                except Exception as exc:
                    logging.error(exc)
                    continue

                loss = trainer.train(df)
                logging.info(f"Epoch {epoch}, File: {file}, Batch {num_batch}, Loss: {loss:.4f}")

        if loss < 0.1:
            logging.info(f"DONE")
            torch.save(model.state_dict(), "model.pth")
            # return model.state_dict()


        val_loss = trainer.validate(validation_data)
        logging.info(f"Epoch {epoch}, Validation loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # best_model_state = model.state_dict()
            torch.save(model.state_dict(), "model.pth")
            logging.info(f"Epoch {epoch}: New best validation loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logging.info(f"Early stopping triggered! No improvement for {patience} epochs.")
                break

        trainer.send_result(telegram=True, local_save=False)
    # return best_model_state


if __name__ == '__main__':
    try:
        best_model_state, filename = train(), "model.pth"
    except Exception as exc:
        logging.error(exc)
    finally:
        send_msg(f'{__file__}: Рассчеты закончены', delete_after=5)
        
    # if best_model_state:
    #     torch.save(best_model_state, filename)
