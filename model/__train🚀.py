import json
import logging
import os
import random
import warnings
from io import StringIO
from os.path import basename, exists, join
from statistics import mean
from typing import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import RMolEncoder as rme
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from __features import *
from _model import ChemLM, Loss, batch_reader, make_fix_len_collate, tokenize


                    ############################################################
                    #                                                          #
                    #                     Мета настройка                       #
                    #                                                          #
                    ############################################################
                    
send_msg(f'{__file__}: Рассчеты начались')

dirname = '/home/lipatovdn/__data__'
if not exists(dirname):
    raise FileNotFoundError(f"Нет папки с обучающей выборкой по пути: {dirname}")

pretrained_path = "pretrained_encoder.pth"
if not exists(pretrained_path):
    raise FileNotFoundError(f"Предобученный файл не найден по пути: {pretrained_path}")

log_filename = 'training.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
if exists(log_filename):
    with open(log_filename, 'w'):
        pass

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=FutureWarning)

def take_dict(filename):
    if exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    raise FileNotFoundError(f"Нет папки с обучающей выборкой по пути: {dirname}")

metafilename = 'data-info.json'
files_dict: dict = take_dict(metafilename)
metaline_dict: dict = take_dict('lineinfo.json')
already_prepared = sum(files_dict.values())
batch_size = 256

def filestatus(file, status=True):
    files_dict[file] = status
    with open(metafilename, 'w') as f:
        json.dump(files_dict, f)



                    ############################################################
                    #                                                          #
                    #                     Класс тренера                        #
                    #                                                          #
                    ############################################################
                    
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
        self.base_lr = 0.001
        
        self.total_count_lines = sum(metaline_dict.values())
        self.total_steps = (self.total_count_lines // batch_size)
        self.current_step = sum(metaline_dict[file] for file, prepared in files_dict.items() if prepared) // 256

        for param_group in self.optim.param_groups:
            param_group['lr'] = 0.0


    def update_lr(self, decay_percentage=0.2, min_lr=1e-4):
        """Learning rate: linear warmup, then exponential decay."""
        if self.current_step < self.warmup_steps:
            return self.base_lr * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            adjusted_progress = progress ** (1 / (1 + decay_percentage))
            lr = min_lr + (self.base_lr - min_lr) * (1 + np.cos(np.pi * adjusted_progress)) * 0.5
            
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




                    ############################################################
                    #                                                          #
                    #                        Обучение                          #
                    #                                                          #
                    ############################################################
                    
def train_epoch(model: ChemLM, files: List[str], batch_size: int, epoch: int, trainer: Trainer):
    losses = []
    random.shuffle(files)
    for num_file, file in enumerate(files, start=1):
        with tqdm(total=count_lines(file), desc=meta(), dynamic_ncols=True) as tq:
            for num_batch, batch in enumerate(batch_reader(file, batch_size), start=1):
                with Timer(name=f'file:{file}, num batch: {num_batch}', logger=logger) as time:
                    try:
                        df: pd.DataFrame = pd.read_csv(StringIO(batch), delimiter=';', encoding='utf-8').dropna()
                    except Exception as exc:
                        logging.error(exc)
                        continue

                    loss: float = trainer.train(df)
                    losses.append(loss)
                    logging.info(
                        f"Epoch {epoch}, File: {basename(file)}, Batch {num_batch:>4}, Loss: {loss:.4f}, MeanLoss: {mean(losses):.4f}, BatchTime: {time}")
                    tq.set_description(f'{meta()}, File {already_prepared+num_file}/{already_prepared+len(files)}')
                    tq.update(batch_size)
        torch.save(model.state_dict(), f"model_tmp.pth")
        filestatus(file)
    return losses


def validate_and_save(model: ChemLM, trainer: Trainer, validation_data: pd.DataFrame, epoch: int,
                      best_val_loss: float, patience: int, epochs_without_improvement: int) -> Tuple[int, float]:
    val_loss = trainer.validate(validation_data)
    logging.info(f"Epoch {epoch}, Validation loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), f"model_{epoch}.pth")
        logging.info(f"Epoch {epoch}: New best validation loss: {best_val_loss:.4f}")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            logging.info(f"Early stopping triggered! No improvement for {patience} epochs.")
    return best_val_loss, epochs_without_improvement


def train(patience: int = 3, batch_size: int = 256, num_epoch: int = 3,
          pretrained_path: Optional[str] = None, dirname: Optional[str] = None) -> None:
    namemodel = 'model_tmp.pth'
    if not exists(namemodel):
        raise FileNotFoundError(f"Нет дампа модели по пути {dirname}")
    model = ChemLM(chem_encoder_params=dict(d_model=512, n_in_head=8, num_in_layers=8, shared_weights=True),
                   decoder_params=dict(d_model=512, nhead=8, dim_feedforward=4 * 512, dropout=0.1,
                                       activation=nn.functional.gelu, batch_first=True, norm_first=True, bias=True),
                   num_layers=4, vocab_size=128, chem_encoder_pretrained_path=pretrained_path)
    model.load_state_dict(torch.load(namemodel, map_location=device))
    logging.info('Model successfully loaded!')
    logging.info(f'We continue from {already_prepared+1} file')
    model.train()
    files = [file for file, processed in files_dict.items() if not processed]
    validation_data: pd.DataFrame = pd.read_csv(join('__val__', 'validation_Compound_000000001_000500000.csv'),
                                                delimiter=';', encoding='utf-8').dropna()
    threshold = 0.1
    best_val_loss = float('inf')
    float('inf')
    epochs_without_improvement = 0
    trainer = Trainer(model)
    logging.info(f'trainer current step = {trainer.current_step}') # THERE NOW
    for epoch in range(1, num_epoch):
        losses = train_epoch(model, files, batch_size, epoch, trainer)
        mean(losses)
        # if mean_loss < min(threshold, past_loss):
        #     past_loss = mean_loss
        torch.save(model.state_dict(), f"model_{epoch}.pth")
        # logging.info("BETTER!")

        best_val_loss, epochs_without_improvement = validate_and_save(model, trainer, validation_data,
                                                                      epoch, best_val_loss, patience, epochs_without_improvement)
        if (epochs_without_improvement >= patience) or (best_val_loss < threshold):
            torch.save(model.state_dict(), f"model_{epoch}.pth")
            break
        trainer.send_result(telegram=True, local_save=False)



                    ############################################################
                    #                                                          #
                    #                       Точка входа                        #
                    #                                                          #
                    ############################################################


if __name__ == '__main__':
    torch.cuda.empty_cache()
    max_retries = 10  
    for retry_count in range(1, max_retries + 1):
        try:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            best_model_state = train(pretrained_path=pretrained_path, dirname=dirname)
            send_msg(f'{__file__}: Рассчеты успешно завершены')
            break  
        except Exception as exc:
            base = basename(__file__)
            logging.error(f"Attempt {retry_count}/{max_retries}. Error: {err_wrap(base, exc)}")
            send_msg(f'{base}: Ошибка (попытка {retry_count}/{max_retries}). Перезапуск...')
            time.sleep(10)
        finally:
            torch.cuda.empty_cache()
            if 'best_model_state' in locals():
                del best_model_state
    
    if retry_count >= max_retries:
        send_msg(f'{basename(__file__)}: Достигнуто максимальное количество попыток ({max_retries}). Прекращение работы.')
        