import pandas as pd
from os.path import join, abspath, basename
from io import StringIO
import pickle as pkl, os, sys
import chython

data_pathdir = abspath(join(os.getcwd(), '.'))
uchar_dirname = '__molecular_containers__'

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
        self.mc_list = list()
    
    def compute(self, *, _batch_size=5):
        for batch in batch_reader(self.abs_filepath, batch_size=_batch_size):
            batch_df: pd.DataFrame = pd.read_csv(StringIO(batch),sep=';')
            for smile in batch_df['SMILES'].to_list():
                self.mc_list.append(chython.smiles(smile))
                
        with open(self.totalname, 'wb') as file:
            pkl.dump(self.mc_list, file)

########### ИНТЕРФЕЙС ########### 

from Kiki._send_msg import send_msg
import psutil
from concurrent.futures import ProcessPoolExecutor
def multymain(load=50):
    csv_files = [f for f in os.listdir(data_pathdir) if f.endswith('.csv')]
    csv_files =  csv_files[:5] # THERE
    print(csv_files)
    # current_load = psutil.cpu_percent(interval=1)
    # available_capacity = 100 - current_load
    # target_usage = available_capacity * load / 100
    # num_workers = max(1, min(round(os.cpu_count() * (target_usage / 100)), len(csv_files)))
    # send_msg(f"Используем {num_workers} воркеров (загрузка CPU: {current_load}%)", delete_after=5)
    # with ProcessPoolExecutor(max_workers=num_workers) as executor:
    #     executor.map(lambda f: UCharHandler(f).compute(), csv_files)


if __name__ == "__main__":
    try:
        send_msg(f'Старт {basename(__file__)}', delete_after=5)
        multymain(load=50) # THERE
        send_msg(f'Конец {basename(__file__)}')
    except Exception as e:
        import traceback
        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        send_msg(f"Ошибка в {basename(__file__)}:\n{tb_str}")
        print(tb_str)
    
    sys.exit(0)