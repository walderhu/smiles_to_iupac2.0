import warnings
from io import StringIO
from os.path import exists
import pandas as pd
import torch
from _model import batch_reader
import logging

from __features import *

dirname = '__train__'
if not exists(dirname):
    raise FileNotFoundError(f"Нет папки с обучающей выборкой по пути: {dirname}")

file = join(dirname, 'train_Compound_000000001_000500000.csv')
size = max_len_file(file)
print(size)