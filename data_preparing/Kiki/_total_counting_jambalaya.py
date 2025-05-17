import pickle, os
import pandas as pd
from os.path import basename, join
from io import StringIO
from collections import Counter
from chython import smiles
from _logger import Logger
from _notification import send_msg

name_dir_archs = '__data__'
path_dir_archs = join(os.getcwd(), name_dir_archs)
name_dir_log = '__total_logging__'
dirname_couner = '__total_counting__'
os.makedirs(name_dir_log, exist_ok=True)
os.makedirs(dirname_couner, exist_ok=True)
_debug=True

    
class Handler:
    def __init__(self, arch):
        self.total_path_to_arch = join(path_dir_archs, arch)
        self.basename = basename(arch.removesuffix('.csv'))
        self.name_logfile = basename(f"{self.basename}.log")
        self.total_path_to_logfile = join(os.getcwd(), name_dir_log, self.name_logfile)
        self.logger = Logger(debug=_debug,log_file=self.total_path_to_logfile,namelog=self.basename)
        self.counter_filename = basename(f"{self.basename}.pkl")
        self.total_counter_filename = join(os.getcwd(), dirname_couner, self.counter_filename)
        self.counter = Counter()        
        
        
    def batch_reader(self, file_path, batch_size=500):
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline().rstrip('\n')  # читаем заголовок один раз
            batch = []
            for line in f:
                batch.append(line.rstrip('\n'))
                if len(batch) == batch_size:
                    yield '\n'.join([header] + batch)
                    batch = []
            if batch: # остаток строк
                yield '\n'.join([header] + batch)
        
            
    def compute(self, *, _batch_size=100):
        for batch in self.batch_reader(self.total_path_to_arch, batch_size=_batch_size):
            df = pd.read_csv(StringIO(batch),sep=';')
            unique_ids = df['cid'].unique()  
            self.counter['All'] += len(unique_ids)

        with open(self.total_counter_filename, 'wb') as file:
            pickle.dump(self.counter, file)
        

def main():
    csv_files = filter(lambda f: f.endswith('.csv'), os.listdir(path_dir_archs))
    logger = Logger(debug=True)
    count = 0
    for file in csv_files:
        try:
            obj = Handler(file)
            obj.compute(_batch_size=50000)
            count += 1
        except Exception as e:
            logger.error(e)
    return count
        

def process_file(file):
    try:
        obj = Handler(file)
        obj.compute(_batch_size=50000)
        return (file, True, None)  
    except Exception as e:
        return (file, False, str(e)) 


def multymain(load):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    csv_files = [f for f in os.listdir(path_dir_archs) if f.endswith('.csv')]
    logger = Logger(debug=True)
    num_workers = round(os.cpu_count() * (load / 100))
    num_workers = max(1, num_workers)
    success_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in csv_files}
        for future in as_completed(futures):
            file = futures[future]
            try:
                file, success, error = future.result()
                if success:
                    success_count += 1
                else:
                    logger.error(f"Error processing {file}: {error}")
            except Exception as e:
                logger.error(f"Unhandled exception for {file}: {e}")
    return success_count


if __name__ == "__main__":
    send_msg(f"Начало подсчета")
    load = 70
    count = multymain(load)
    send_msg(f"Успешный подсчет {count} файлов")
    os._exit(0)
