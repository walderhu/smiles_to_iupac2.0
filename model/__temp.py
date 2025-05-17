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

dirname = '.'

files = [join(dirname, f) for f in os.listdir(dirname) if f.endswith('.csv') and not f.startswith('_')]
print(files)