from tqdm import tqdm
import operator
from functools import reduce
from concurrent.futures import ProcessPoolExecutor
import psutil
import csv
import os
import pandas as pd
import pickle
from pandas import DataFrame
from Kiki._send_msg import send_msg
from collections import Counter
from os.path import join, abspath, basename
from io import StringIO

data_pathdir = '__unique_data__'
copy_pathdir = '__counting__'
newdata_pathdir = '__data__'

os.makedirs(newdata_pathdir, exist_ok=True)
os.makedirs(copy_pathdir, exist_ok=True)


class Handler:
    def __init__(self, filepath: str):
        self.counter = Counter()
        self.absfilepath = abspath(join(data_pathdir, filepath))
        self.new_filepath = abspath(join(newdata_pathdir, filepath))
        counter_filename = basename(f"{filepath.removesuffix('.csv')}.pkl")
        self.counter_absfilename = abspath(
            join(copy_pathdir, counter_filename))

    def batch_reader(self, file_path, batch_size):
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline().rstrip('\n')
            batch = []
            for line in f:
                batch.append(line.rstrip('\n'))
                if len(batch) == batch_size:
                    yield '\n'.join([header] + batch)
                    batch = []
            if batch:
                yield '\n'.join([header] + batch)

    def filter(self, raw_df: DataFrame) -> DataFrame:
        filtered_df = raw_df[raw_df['SMILES'].notna()]
        self.counter['None'] += len(raw_df) - len(filtered_df)
        whole_df = filtered_df.drop_duplicates(subset=['SMILES'], keep='first')
        self.counter['Duplicates'] += len(filtered_df) - len(whole_df)
        return whole_df

    def execute(self):
        _batch_size = 50000
        with open(self.new_filepath, mode='w', encoding='utf-8') as f:
            _header = True
            for batch in self.batch_reader(
                    self.absfilepath, batch_size=_batch_size):
                try:
                    batch_df = pd.read_csv(StringIO(batch), sep=';')
                    whole_df: DataFrame = self.filter(batch_df)
                    df = whole_df.to_csv(
                        sep=';', index=False, header=_header, quoting=csv.QUOTE_ALL)
                    _header = False
                    f.write(df)
                    self.counter['Correct'] += 1
                except Exception as e:
                    self.counter['Incorrect'] += 1
                finally:
                    self.counter['All'] += 1

        with open(self.counter_absfilename, 'wb') as file:
            pickle.dump(self.counter, file)


def debug():
    import traceback
    file = [f for f in os.listdir(data_pathdir) if f.endswith('.csv')][0]
    print(f'{file=}')
    try:
        Handler(file).execute()
    except Exception as e:
        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        print(tb_str)
        os._exit(0)

def execute(file): return Handler(file).execute()

def multymain(load=50):
    csv_files = [f for f in os.listdir(data_pathdir) if f.endswith('.csv')]
    current_load = psutil.cpu_percent(interval=1)
    target_usage = ((100 - current_load) * load) / 100
    num_workers = max(1, min(round(os.cpu_count() * (target_usage / 100)), len(csv_files)))
    send_msg(f"Используем {num_workers} воркеров (загрузка CPU: {current_load}%)", delete_after=5)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(execute, csv_files))


def counter() -> Counter:
    files = (join(copy_pathdir, f) for f in os.listdir(copy_pathdir))
    counters = (pickle.load(open(file, "rb"))
                for file in files if file.endswith('.pkl'))
    return reduce(operator.add, counters, Counter())


if __name__ == "__main__":
    load = 70
    send_msg('Начало удаления дубликатов', delete_after=5)
    multymain(load)  # THERE
    count = counter()
    send_msg(f'Конец расчетов.\n{count}')
    os._exit(0)
