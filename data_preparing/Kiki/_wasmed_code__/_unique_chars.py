import pandas as pd
from os.path import join, abspath, basename
from io import StringIO
import pickle as pkl, os, sys


data_pathdir = abspath(join(os.getcwd(), '__unique_data__'))
uchar_dirname = '__uchar_data__'

if not os.path.exists(uchar_dirname):
    os.makedirs(uchar_dirname)

########### ЯДРО ########### 

def batch_reader(file_path, batch_size):
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
            
class UCharHandler:
    def __init__(self, filepath):
        self.abs_filepath = abspath(join(data_pathdir, filepath))
        self.uchar_filename = basename(f"{filepath.removesuffix('.csv')}.pkl")
        self.totalname = join(uchar_dirname, self.uchar_filename)
        self.uchars = set()
    
    def compute(self, *, _batch_size=5):
        for batch in batch_reader(self.abs_filepath, batch_size=_batch_size):
            batch_df: pd.DataFrame = pd.read_csv(StringIO(batch),sep=';')
            for smile in batch_df['SMILES'].dropna().astype(str):
                self.uchars.update(smile)
                
        with open(self.totalname, 'wb') as file:
            pkl.dump(self.uchars, file)

########### ИНТЕРФЕЙС ########### 
from tqdm import tqdm
from Kiki._send_msg import send_msg
import psutil
from concurrent.futures import ProcessPoolExecutor

def process_file(f):
    obj = UCharHandler(f)
    obj.compute()
    
def multymain(load=50):
    csv_files = [f for f in os.listdir(data_pathdir) if f.endswith('.csv')]
    current_load = psutil.cpu_percent(interval=1)
    available_capacity = 100 - current_load
    target_usage = available_capacity * load / 100
    num_workers = max(1, min(round(os.cpu_count() * (target_usage / 100)), len(csv_files)))
    send_msg(f"Используем {num_workers} воркеров (загрузка CPU: {current_load}%)", delete_after=5)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_file, csv_files), total=len(csv_files)))


def check_data():
    pkl_files = [join(uchar_dirname, f) for f in os.listdir(uchar_dirname) if f.endswith('.pkl')]
    result = set().union(*(pkl.load(open(f, 'rb')) for f in pkl_files))
    print(result)

if __name__ == "__main__":
    check = True
    try:
        if check:
            check_data()
        else:
            send_msg(f'Старт {basename(__file__)}', delete_after=5)
            # multymain(load=80) 
            send_msg(f'Конец {basename(__file__)}')
    except Exception as e:
        import traceback
        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        send_msg(f"Ошибка в {basename(__file__)}:\n{tb_str}")
        print(tb_str)
    sys.exit(0)