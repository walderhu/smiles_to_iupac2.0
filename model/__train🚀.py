from statistics import mean
import logging
import os
import random
import warnings
from io import StringIO
from os.path import exists, join, basename
from __features import err_wrap
from tqdm import tqdm
from typing import *
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
        
        # Настройки для аккумуляции градиентов
        self.virtual_batch_size = 256
        self.batch_size = 32
        self.grad_accum = self.virtual_batch_size // self.batch_size
        assert self.virtual_batch_size % self.batch_size == 0, "virtual_batch_size must be divisible by batch_size"
        self.accum_step = 0
        self.accum_losses = []
        
        # Warmup параметры
        self.warmup_steps = 800  
        self.current_step = 0    
        self.base_lr = 0.001     
        
        # Инициализация начального learning rate
        for param_group in self.optim.param_groups:
            param_group['lr'] = 0.0 

    def update_lr(self):
        """Обновляет learning rate согласно расписанию warmup"""
        if self.current_step < self.warmup_steps:
            # Линейное увеличение lr в течение warmup периода
            lr = self.base_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr
        self.current_step += 1

    def train(self, df: pd.DataFrame) -> float:
        MAX_SEQ_LEN = df["IUPAC Name"].apply(len).max()
        DS_SEQ_LEN = MAX_SEQ_LEN + 3
        smis = list(df['SMILES'])
        seqs = [tokenize(df.iloc[i]["IUPAC Name"], add_spezial_tokens=True) for i in range(df.shape[0])]

        ds = rme.dataset.RxnMolDataset(smis, seqs)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size,
                                      collate_fn=make_fix_len_collate(DS_SEQ_LEN), 
                                      shuffle=True, drop_last=False)
        losses = []
        for b in dl:
            x, y = b
            x, y = x.to(device), y.to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                p = self.model(x, y[:, :-1])
                loss = Loss(p, y[:, 1:])
                loss = loss / self.grad_accum

            self.accum_losses.append(float(loss) * self.grad_accum)
            self.scaler.scale(loss).backward()
            
            self.accum_step += 1
            if self.accum_step % self.grad_accum == 0:
                # Обновляем learning rate перед шагом оптимизации
                self.update_lr()
                
                # Применяем clipping градиентов
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad()
                
                avg_loss = np.mean(self.accum_losses)
                losses.append(avg_loss)
                self.accum_losses = []

        self.losses.extend(losses)
        return np.mean(losses) if losses else 0.0
    
    
    
    

    def validate(self, df: pd.DataFrame):
        self.model.eval()
        MAX_SEQ_LEN = df["IUPAC Name"].apply(len).max()
        DS_SEQ_LEN = MAX_SEQ_LEN + 3
        smis = list(df['SMILES'])
        seqs = [tokenize(df.iloc[i]["IUPAC Name"], add_spezial_tokens=True) for i in range(df.shape[0])]
        ds = rme.dataset.RxnMolDataset(smis, seqs)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, 
                                        collate_fn=make_fix_len_collate(DS_SEQ_LEN),
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


                
def train_epoch(files: List[str], batch_size: int, epoch: int, trainer: Trainer):
    losses = []
    random.shuffle(files)
    for file in files:
        with tqdm(total=count_lines(file), desc=meta(), dynamic_ncols=True) as tq:
            for num_batch, batch in enumerate(batch_reader(file, batch_size), start=1):
                try:
                    df: pd.DataFrame = pd.read_csv(StringIO(batch), delimiter=';', encoding='utf-8').dropna()
                except Exception as exc:
                    logging.error(exc)
                    continue

                loss: float = trainer.train(df)
                losses.append(loss)
                logging.info(f"Epoch {epoch}, File: {basename(file)}, Batch {num_batch}, Loss: {loss:.4f}, MeanLoss: {mean(losses):.4f}")
                tq.set_description(meta())
                tq.update(batch_size)
    return losses


def validate_and_save(model: rme, trainer: Trainer, validation_data: pd.DataFrame, epoch: int, \
                best_val_loss: float, patience: int, epochs_without_improvement: int) -> Tuple[int, float]:
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


def train(patience:int=3, batch_size:int=256, num_epoch:int=10, \
                pretrained_path:Optional[str]=None, dirname:Optional[str]=None) -> None:  
    
    model = ChemLM(chem_encoder_params=dict(d_model=512, n_in_head=8, num_in_layers=8, shared_weights=True),
                   decoder_params=dict(d_model=512, nhead=8, dim_feedforward=4 * 512, dropout=0.1,
                                       activation=nn.functional.gelu, batch_first=True, norm_first=True, bias=True),
                   num_layers=4, vocab_size=128, chem_encoder_pretrained_path=pretrained_path)
    model.train()
    files = [join(dirname, 'train_Compound_000000001_000500000.csv')] 
    validation_data: pd.DataFrame = pd.read_csv(join('__val__', 'validation_Compound_000000001_000500000.csv'),
                                 delimiter=';', encoding='utf-8').dropna()
    threshold = 0.1
    best_val_loss = float('inf')
    past_loss = float('inf')
    epochs_without_improvement = 0
    trainer = Trainer(model) 
    for epoch in range(1, num_epoch):
        losses = train_epoch(files, batch_size, epoch, trainer)
        mean_loss = mean(losses)
        if mean_loss < min(threshold, past_loss):
            past_loss = mean_loss
            torch.save(model.state_dict(), f"model_{epoch}.pth") 
            logging.info("BETTER!")

        best_val_loss, epochs_without_improvement = validate_and_save(model, trainer, validation_data, \
                                            epoch, best_val_loss, patience, epochs_without_improvement)
        if (epochs_without_improvement >= patience) or (best_val_loss < threshold):
            torch.save(model.state_dict(), f"model_{epoch}.pth") 
            break
        trainer.send_result(telegram=True, local_save=False)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    try:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        best_model_state, filename = train(pretrained_path=pretrained_path, dirname=dirname), "model.pth"
        send_msg(f'{__file__}: Рассчеты успешно завершены', delete_after=5)
    except Exception as exc:
        base = basename(__file__)
        logging.error(err_wrap(base, exc))
        send_msg(f'{base}: Рассчеты закончены c ошибкой', delete_after=5)
    finally:
        torch.cuda.empty_cache()
        if 'best_model_state' in locals():
            del best_model_state