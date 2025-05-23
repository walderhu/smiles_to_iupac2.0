import logging
import os
import tempfile
from io import StringIO
from os.path import basename, exists, join

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
LOGFILENAME = 'training.log'
VOCAB_SIZE = 500
BATCH_SIZE = 512
os.makedirs(TOKEN_DIRNAME, exist_ok=True)

logging.basicConfig(filename=LOGFILENAME, level=logging.INFO, format='%(message)s')
if exists(LOGFILENAME):
    with open(LOGFILENAME, 'w'):
        pass

#            ############################################################            #
#            #                                                          #            #
#            #                         Обработка                        #            #
#            #                                                          #            #
#            ############################################################            #


def handler(filepath: str):
    filename: str = basename(filepath)
    model_filenpath = join(TOKEN_DIRNAME, filename.removesuffix('.csv'))
    with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
        for num_batch, batch in enumerate(batch_reader(filepath, BATCH_SIZE), start=1):
            if num_batch == 5:
                break
            batch_df = pd.read_csv(StringIO(batch), sep=';')
            smiles_batch = batch_df['IUPAC Name'].dropna().tolist()
            f.write("\n".join(smiles_batch) + "\n")

        f.flush()
        SentencePieceTrainer.train(
            input=f.name,
            model_prefix=model_filenpath,
            vocab_size=VOCAB_SIZE,
            model_type="bpe",
            split_digits=True,
            treat_whitespace_as_suffix=False,
            normalization_rule_name="identity"
        )

#            ############################################################            #
#            #                                                          #            #
#            #                       Точка входа                        #            #
#            #                                                          #            #
#            ############################################################            #


def main():
    files = [join(DATA_DIRNAME, f) for f in os.listdir(DATA_DIRNAME) if f.endswith('.csv')]
    print(len(files))
    for file in files:
        handler(file)
        break


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        base = basename(__file__)
        logging.error(f"Error: {err_wrap(base, exc)}")
        print(f"Error: {err_wrap(base, exc)}")
