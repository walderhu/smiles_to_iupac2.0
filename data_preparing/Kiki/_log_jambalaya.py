import pickle
import logging
import os, sys
import pandas as pd
from io import StringIO
from collections import Counter
from chython import smiles



name_dir_archs = '__gz_to_csv_data__'
path_dir_archs = os.path.join(os.getcwd(), name_dir_archs)
name_dir_log = '__logging3__'
dirname_couner = '__counting3__'
os.makedirs(name_dir_log, exist_ok=True)
os.makedirs(dirname_couner, exist_ok=True)
_debug=True


class Logger:     
    def __init__(self, *, _debug=False, _log_file='app.log', _namelog=__name__):
        _format = '%(message)s [File: "%(filename)s", line %(lineno)d]' if _debug else '%(message)s'
        _logger = logging.getLogger(_namelog)
        _logger.setLevel(logging.DEBUG)  
        file_formatter = logging.Formatter(_format)
        console_formatter = logging.Formatter('%(message)s')

        file_handler = logging.FileHandler(_log_file)
        file_handler.setLevel(logging.DEBUG)  
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(self.Filter())

        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)  
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(self.Filter(equal=False))

        _logger.addHandler(file_handler)
        _logger.addHandler(console_handler)
        self._logger = _logger

    class Filter(logging.Filter):
        def __init__(self, name = "", *, level=logging.INFO, equal=True):
            super().__init__(name)
            self.level, self.equal = level, equal

        def filter(self, record):
            return (record.levelno == self.level) == self.equal

    def __getattr__(self, item):
        return getattr(self._logger, item)



from chython.algorithms.standardize._groups import _rules_single, _rules_double 
class TautomerHandler:
    def __init__(self):
        self.rules: list = _rules_single() + _rules_double()
        self.rules_dict = {str(rule[0]): rule[4] for rule in self.rules}

    def get_status(self, canonocalized_log: str) -> bool:
        try:
            info_rul = canonocalized_log[0][2]
            result = self.rules_dict[info_rul]
            return result
        except:
            return False



class Handler:
    TauHandler = TautomerHandler()
    
    def __init__(self, arch):
        self.total_path_to_arch = os.path.join(path_dir_archs, arch)
        self.basename = os.path.basename(arch.removesuffix('.csv'))
        self.name_logfile = os.path.basename(f"{self.basename}.log")
        self.total_path_to_logfile = os.path.join(os.getcwd(), name_dir_log, self.name_logfile)
        self.logger = Logger(_debug=_debug,_log_file=self.total_path_to_logfile,_namelog=self.basename)
        keys = ['Inorganic', 'Salt', 'Canonicalize', 'Tautomer', 'Copy', 'Radical', 'BadSmiles', 'Isotop', 'BadCanonicalize', 'Neutralize', 'Correct']
        self.counter = Counter({key: 0 for key in keys})        
        self.is_taytomer = self.TauHandler.get_status
        
        self.counter_filename = os.path.basename(f"{self.basename}.pkl")
        self.total_counter_filename = os.path.join(os.getcwd(), dirname_couner, self.counter_filename)
        
        
    def batch_reader(self, file_path, batch_size=500):
        """
        Генератор, который читает файл и возвращает батчи по batch_size строк
        в виде многострочной строки, включая заголовок первой строкой.
        """
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
        
            
    def compute(self):
        _batch_size = 10000 
        for batch in self.batch_reader(self.total_path_to_arch, batch_size=_batch_size):
            df = pd.read_csv(StringIO(batch),sep=';')
            self.copy_handler(df)
            smiles = df['PUBCHEM_SMILES']
            ids = df['PUBCHEM_COMPOUND_CID']
            ids = [i for i in range(len(smiles))]
            for id, smile in zip(ids, smiles):
                self.counter['All'] += 1
                try:
                    self.smile_processing(id, smile)
                    self.counter['Correct'] += 1
                except Exception as e:
                    self.counter['Incorrect'] += 1

        with open(self.total_counter_filename, 'wb') as file:
            pickle.dump(self.counter, file)
        
        
    def copy_handler(self, df):
        lst = df['STANDARDIZE_CHYTHON'].tolist()
        self.counter['Copy'] += (len(lst) - len(set(lst)))
            

    def smile_processing(self, id, smile):
        """
        Функция рассчитывает наличие у SMILES'а принадлежность к металлорганике/неорганике,
        солям, или к форме молекулы с не правильно записанным хенотипом
        """
        m = smiles(smile, ignore=True, remap=False, 
            ignore_stereo=False, ignore_bad_isotopes=False,
            keep_implicit=True, ignore_carbon_radicals=False,
            ignore_aromatic_radicals=False)
        if not m:
            self.counter['BadSmiles'] += 1
            raise Exception('Не удалось прочитать')
        
        if m.is_radical:
            self.counter['Radical'] += 1
            raise Exception('Радикал')

        if m.clean_isotopes():
            self.counter['Isotop'] += 1
        
        unique_atoms = set([a[1].atomic_symbol for a in list(m.atoms())])  # уникальные атомы для проверки на то, что соединение органическое
        if len(unique_atoms.intersection(set(['c', 'C']))) == 0 or \
           len(unique_atoms.intersection(self.METAL_LIST)) > 0:
            self.counter['Inorganic'] += 1
        try:
            canonocalized_log = m.canonicalize(fix_tautomers=True, logging=True)
            if self.is_taytomer(canonocalized_log):
                self.counter['Tautomer'] += 1

            self.logger.info(f'{id}: {canonocalized_log}')                
            if m.neutralize():
                self.counter['Neutralize'] += 1
        except:
            self.counter['BadCanonicalize'] += 1
            raise Exception('Неуспешная канонизация')

        if self.salt_handler(m):
            self.counter['Salt'] += 1
            raise Exception('Соли')
         
        if smile != str(m):
            self.counter['Canonicalize'] += 1
            
            
    METAL_LIST = {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Ga', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Cd', 'In', 'Sn',
                'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'Ac', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu',
                'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Ge', 'Sb'}

    def salt_handler(self, m):                      
        return m.split_metal_salts(logging=True) or \
           m.connected_components_count > 1
            

def main():
    archs = [f for f in os.listdir(path_dir_archs) if f.endswith('.csv')] # csv_files
    for arch in archs:
        try:
            obj = Handler(arch)
            obj.compute()
        except Exception as e:
            RED, RESET = "\033[31m", "\033[0m"
            _format = RED + 'Exception: ' + RESET + '%(message)s [File: "%(filename)s", line %(lineno)d]'
            logging.basicConfig(level=logging.ERROR, format=_format)
            logging.error(e)
        
    
def wrapper(file):
    try:
        obj = Handler(file)
        obj.compute()
    except Exception as e:
        RED, RESET = "\033[31m", "\033[0m"
        _format = RED + 'Exception: ' + RESET + '%(message)s [File: "%(filename)s", line %(lineno)d]'
        logging.basicConfig(level=logging.ERROR, format=_format, 
            filename='error.log', filemode='a')
        logging.error(e, exc_info=True)  

        
def multymain():
    from concurrent.futures import ProcessPoolExecutor
    files = [f for f in os.listdir(path_dir_archs) if f.endswith('.csv')] 
    percentage_workload = 70
    num_workers = round(os.cpu_count() * (percentage_workload / 100))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(wrapper, files)
        
        
def send_msg(text):
    import telebot, json
    with open('__property.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    bot = telebot.TeleBot(data['token'])
    bot.send_message(data['CHAT_ID'], text)
    
    
if __name__ == '__main__':
    multymain()
    send_msg('Рассчеты завершены')
    os._exit(0)
    
    
    