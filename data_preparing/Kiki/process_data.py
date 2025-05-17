import pandas as pd
import xml.etree.ElementTree as ET
import gzip, csv, os
from os.path import join, basename
from _notification import send_msg
from _logger import Logger

from chython import smiles
from collections import Counter
import pickle
from pprint import pprint


_debug = True
xml_pathdir = '__xml_data__'
csv_pathdir = '__data__'
name_dir_log = '__log__'
dirname_couner = '__count__'


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
    start_butch = '<PC-Compound>'
    end_butch = '</PC-Compound>'
    TauHandler = TautomerHandler()

    METAL_LIST = {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Ga', 
                  'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Cd', 'In',
                  'Sn', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
                  'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'Ac', 'Ce', 'Pr', 'Nd', 'Pm',
                  'Sm', 'Eu','Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                  'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es',
                  'Fm', 'Md', 'No', 'Lr', 'Ge', 'Sb'}

    def __new__(cls, *args, **kwargs):
        os.makedirs(csv_pathdir, exist_ok=True)
        os.makedirs(name_dir_log, exist_ok=True)
        os.makedirs(dirname_couner, exist_ok=True)
        return super().__new__(cls)


    def __init__(self, xml_file):
        self.xml_file = xml_file
        self.xml_path = join(xml_pathdir, self.xml_file)
        
        self.basename = os.path.basename(self.xml_file.removesuffix('.xml.gz'))
        self.csv_file = basename(f"{self.basename}.csv")
        self.csv_path = join(csv_pathdir, self.csv_file)

        self.name_logfile = os.path.basename(f"{self.basename}.log")
        self.total_path_to_logfile = os.path.join(os.getcwd(), name_dir_log, self.name_logfile)
        
        keys = ['Inorganic', 'Salt', 'Canonicalize',
                'Tautomer', 'Copy', 'Radical',
                'BadSmiles', 'Isotop', 'BadCanonicalize',
                'Neutralize', 'Correct' ]
        self.counter = Counter({key: 0 for key in keys})        
        self.is_taytomer = self.TauHandler.get_status

        self.counter_filename = os.path.basename(f"{self.basename}.pkl")
        self.total_counter_filename = os.path.join(os.getcwd(), dirname_couner, self.counter_filename)

        
    def read_batches(self, xml_file):
        with gzip.open(xml_file, encoding='utf-8', mode='rt') as f:
            begin = True
            in_batch = False
            buffer = ""
            for line in f:
                if begin and self.start_butch not in line:
                    continue
                buffer += line
                if self.start_butch in line:
                    begin = False
                    in_batch = True
                if self.end_butch in line and in_batch:
                    in_batch = False
                    yield buffer  
                    buffer = ""  

            if buffer and in_batch:
                yield buffer


    def get(self, data, level, *, begin='.//PC-Urn_'):
        result = level.find(f'{begin}{data}')
        return result.text if result is not None else None
        
    
    
    def process(self, batch: str):
        compress_data = {'cid': None, 'IUPAC Name': 'Preferred', 'SMILES': 'Absolute' }
        root = ET.fromstring(batch)
        cid = root.find('.//PC-CompoundType_id_cid').text
        props = root.find('.//PC-Compound_props')
        info_data = props.findall('.//PC-InfoData')
        first_values = {'cid': cid}  

        for data in info_data:
            urn = data.find('.//PC-Urn')
            if urn is None:
                continue
            label = self.get('label', level=urn)
            name = self.get('name', level=urn)  
            
            if label not in compress_data:
                continue
            if compress_data[label] is not None and name != compress_data[label]:
                continue  # Пропускаем, если name не соответствует (например, не Absolute SMILES)
            if label in first_values:
                continue  # Уже записано
            
            value = self.get('.//PC-InfoData_value_sval', level=data, begin='')
            first_values[label] = value
            if all(k in first_values for k in compress_data):
                break
        
        utility_df = pd.DataFrame([first_values])
        utility_df['SMILES'] = utility_df['SMILES'].apply(self.smiles_prepare)
        return utility_df
    
    
    def smiles_prepare(self, smile):
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
        
        unique_atoms = set([a[1].atomic_symbol for a in list(m.atoms())])  
        if len(unique_atoms.intersection(set(['c', 'C']))) == 0 or \
           len(unique_atoms.intersection(self.METAL_LIST)) > 0:
            self.counter['Inorganic'] += 1
        try:
            canonocalized_log = m.canonicalize(fix_tautomers=True, logging=True)
            if self.is_taytomer(canonocalized_log):
                self.counter['Tautomer'] += 1
            if m.neutralize():
                self.counter['Neutralize'] += 1
        except:
            self.counter['BadCanonicalize'] += 1
            raise Exception('Неуспешная канонизация')

        if m.split_metal_salts(logging=True) or \
           m.connected_components_count > 1:
            self.counter['Salt'] += 1
            raise Exception('Соли')
        if smile != str(m):
            self.counter['Canonicalize'] += 1
        return str(m)
            

    def compute_debug(self):
        with open(self.csv_path, mode='w', encoding='utf-8') as f:
            _header = True
            for i, batch in enumerate(self.read_batches(self.xml_path), start=1):
                try:
                    utility_df: pd.DataFrame = self.process(batch)
                    df = utility_df.to_csv(sep=';', index=False, header=_header, quoting=csv.QUOTE_ALL)
                    _header = False
                    f.write(df)
                    self.counter['Correct'] += 1 # корректная обработка батча (то есть умножить на размер батча)
                except:
                    self.counter['Incorrect'] += 1
                finally:
                    self.counter['All'] += 1
                                
        with open(self.total_counter_filename, 'wb') as file:
            pickle.dump(self.counter, file)


########### КЛИЕНТ ###########

def process_file(xml_file):
    try:
        obj = Handler(xml_file)
        obj.compute_debug()
    except Exception as e:
        logger = Logger(debug=_debug, error_filename='error_process_xml.log')
        logger.error(e)


def multymain(load):
    from concurrent.futures import ProcessPoolExecutor
    csv_files = [f for f in os.listdir(xml_pathdir) if f.endswith('.xml.gz')]
    num_workers = max(1, round(os.cpu_count() * (load / 100)))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_file, csv_files)


if __name__ == "__main__":
    load = 70
    send_msg('Начало')
    multymain(load)
    send_msg('Конец расчетов')
    os._exit(0)


