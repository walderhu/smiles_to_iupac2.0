import csv, os
import pandas as pd, pickle
from _notification import send_msg
from collections import Counter
from os.path import join, abspath, basename
from io import StringIO
from _logger import Logger

_debug = True
data_pathdir = '__data__' 

copy_pathdir = '__count_copy__'
newdata_pathdir = '__unique_data__'
os.makedirs(newdata_pathdir, exist_ok=True)
os.makedirs(copy_pathdir, exist_ok=True)

class UniqueHandler:
    def __init__(self, filepath):
        self.counter = Counter()
        self.absfilepath = abspath(join(data_pathdir, filepath))
        self.new_filepath = abspath(join(newdata_pathdir, filepath))
        counter_filename = basename(f"{filepath.removesuffix('.csv')}.pkl")
        self.counter_absfilename = abspath(join(copy_pathdir, counter_filename))
    
    def batch_reader(self, file_path, batch_size):
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
                
    def get_unique_data(self, raw_df):
        unique_df = raw_df.drop_duplicates(subset=['SMILES'], keep='first')
        self.counter['Copy'] += len(raw_df) - len(unique_df)
        return unique_df

    def compute(self):
        _batch_size = 50000
        with open(self.new_filepath, mode='w', encoding='utf-8') as f:
            _header = True
            for batch in self.batch_reader(self.absfilepath, batch_size=_batch_size):
                try:
                    batch_df = pd.read_csv(StringIO(batch),sep=';')
                    unique_df: pd.DataFrame = self.get_unique_data(batch_df)
                    df = unique_df.to_csv(sep=';', index=False, header=_header, quoting=csv.QUOTE_ALL)
                    _header = False
                    f.write(df)
                    self.counter['Correct'] += 1 # корретктная обработка батча
                except Exception as e:
                    self.counter['Incorrect'] += 1
                finally:
                    self.counter['All'] += 1

        with open(self.counter_absfilename, 'wb') as file:
            pickle.dump(self.counter, file)
            

def process_file(xml_file):
    try:
        obj = UniqueHandler(xml_file)
        obj.compute()
    except Exception as e:
        logger = Logger(debug=_debug, error_filename='error_process_xml.log')
        logger.error(e)

def debug():
    import traceback
    xml_file = "Compound_145500001_146000000.csv"
    try:
        obj = UniqueHandler(xml_file)
        obj.compute()
    except Exception as e:
        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        print(tb_str)
        os._exit(0)
        
def multymain(load):
    from concurrent.futures import ProcessPoolExecutor
    csv_files = [f for f in os.listdir(data_pathdir) if f.endswith('.csv')]
    num_workers = max(1, round(os.cpu_count() * (load / 100)))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_file, csv_files)


from functools import reduce
import operator
def sum(*counters):
    return reduce(operator.add, counters)

def counter():
    dirname = join(os.getcwd(), copy_pathdir)
    files = [join(dirname, f) for f in os.listdir(dirname) if f.endswith('.pkl')]
    counters = []
    for file in files:
        with open(file, "rb") as f:
            deserialized_data = pickle.load(f)
            counters.append(deserialized_data)

    total: Counter = sum(*counters)
    df = pd.DataFrame(total.items(), columns=['Process', 'Count'])
    df = df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    return '\n'.join((f"{row['Process']}: {row['Count']}" for _, row in df.iterrows()))


if __name__ == "__main__":
    load = 70
    send_msg('Начало удаления дубликатов')
    multymain(load) # THERE
    count = counter()
    send_msg(f'Конец расчетов.\n{count}')
    
    os._exit(0)
