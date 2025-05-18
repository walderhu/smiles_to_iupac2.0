import warnings
from io import StringIO
from os.path import exists, join
import pandas as pd
import torch
from _model import batch_reader
import logging


def log_err_wrap(exc, min_d=85, max_d=105) -> None: 
    message = str(exc) 
    wrapped_message = ""
    current_line = ""
    words = message.split()
    for word in words:
        if len(current_line) + len(word) + 1 <= max_d: 
            current_line += (word + " ")
        else:
            if len(current_line) >= min_d:
                wrapped_message += current_line.rstrip() + "\n"
                current_line = word + " "
            else:
                wrapped_message += current_line.rstrip() + "\n"
                wrapped_message += word + "\n" 
                current_line = ""
    wrapped_message += current_line.rstrip()
    logging.error(wrapped_message)


def max_len_file(file) -> int:
    max_length = 0 
    for batch in batch_reader(file, batch_size=256):  
        df = pd.read_csv(StringIO(batch), delimiter=';', encoding='utf-8').dropna()
        max_length_batch = df['IUPAC Name'].astype(str).str.len().max()
        if max_length_batch > max_length:
            max_length = max_length_batch
    return max_length
