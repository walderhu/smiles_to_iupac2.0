import psutil
import logging
import os
import tempfile
from io import StringIO
from os.path import basename, exists, join
from datetime import datetime, timedelta
import json
from time import sleep
import pandas as pd
from sentencepiece import SentencePieceTrainer

from __features import *

#            ############################################################            #
#            #                                                          #            #
#            #                     Мета настройка                       #            #
#            #                                                          #            #
#            ############################################################            #

DATA_DIRNAME = '__data__'
TOKEN_DIRNAME = '__tokens__'
LOGFILENAME = 'tokenize.log'
VOCAB_SIZE = 500
BATCH_SIZE = 512
cur_load_perc = 50
os.makedirs(TOKEN_DIRNAME, exist_ok=True)
logging.basicConfig(filename=LOGFILENAME, level=logging.INFO, format='%(message)s')
if exists(LOGFILENAME):
    with open(LOGFILENAME, 'w'):
        pass

METAFILE = 'token_prepared-info.json'
if exists(METAFILE):
    with open(METAFILE, 'r') as f:
        files_dict = json.load(f)
else:
    files = [join(DATA_DIRNAME, f) for f in os.listdir(DATA_DIRNAME) if f.endswith('.csv') and not f.startswith('_')]
    files_dict = {file: False for file in files}
    with open(METAFILE, 'w') as f:
        json.dump(files_dict, f)

already_prepared = sum(files_dict.values())
        
        
#            ############################################################            #
#            #                                                          #            #
#            #                         Обработка                        #            #
#            #                                                          #            #
#            ############################################################            #


def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


def update_workload(curr_load, bias=0.1, max_load=90):
    curr_load *= (1 + bias) if psutil.cpu_percent(interval=0.5) < max_load else (1 - bias)
    return clamp(round(curr_load), 1, 100)


def num_workers(max_percent=50):
    current_load = psutil.cpu_percent(interval=0.5)
    available_capacity = 100 - current_load
    target_usage = available_capacity * max_percent / 100
    return max(1, round(os.cpu_count() * (target_usage / 100)))


def GENERETE(file: str):
    filename: str = basename(file)
    model_filenpath = join(TOKEN_DIRNAME, filename.removesuffix('.csv'))
    with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
        for num_batch, batch in enumerate(batch_reader(file, BATCH_SIZE), start=1):
            batch_df = pd.read_csv(StringIO(batch), sep=';')
            smiles_batch = batch_df['IUPAC Name'].dropna().tolist()
            f.write("\n".join(smiles_batch) + "\n")

        f.flush()

        global cur_load_perc
        cur_load_perc = update_workload(cur_load_perc, max_load=90)
        workers = num_workers(cur_load_perc)
        logging.info(f'Use {workers} workers, current load is {cur_load_perc}')
        
        SentencePieceTrainer.train(
            input=f.name,
            model_prefix=model_filenpath,
            vocab_size=VOCAB_SIZE,
            model_type="bpe",
            split_digits=True,
            treat_whitespace_as_suffix=False,
            normalization_rule_name="identity",
            num_threads=workers
        )
        files_dict[file] = True
        with open(METAFILE, 'w') as f:
            json.dump(files_dict, f)

#            ############################################################            #
#            #                                                          #            #
#            #                       Точка входа                        #            #
#            #                                                          #            #
#            ############################################################            #


def main():
    files = [file for file, processed in files_dict.items() if not processed]
    total_files = len(files)
    for num_file, file in enumerate(files, start=1):
        logging.info(f"Prepare File: {basename(file)}, {already_prepared + num_file}/{already_prepared + total_files}")
        GENERETE(file)


if __name__ == "__main__":
    while True:
        try:
            main()
            break
        except Exception as exc:
            new_time = datetime.now() + timedelta(hours=3)  
            time_str = new_time.strftime("%H:%M:%S %d.%m.%Y")
            base = basename(__file__)
            errmsg = f"Error: {err_wrap(base, exc)}\n🕛: {time_str}"
            logging.error(errmsg)
            print(errmsg)
            sleep(10)