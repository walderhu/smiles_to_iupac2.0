import pandas as pd
import xml.etree.ElementTree as ET
import gzip, csv, os
from os.path import join, basename
from _notification import send_msg
from _logger import Logger

from chython import smiles
from collections import Counter
import pickle
from chython.algorithms.standardize._groups import _rules_single, _rules_double 



_debug = True
logger = Logger(debug=_debug, error_filename='error_process_xml.log')
csv_pathdir = '__data__'
xml_pathdir = '__xml_data__'
os.makedirs(csv_pathdir, exist_ok=True)
start_butch = '<PC-Compound>'
end_butch = '</PC-Compound>'

def read_batches(xml_file):
    with gzip.open(xml_file, encoding='utf-8', mode='rt') as f:
        begin = True
        in_batch = False
        buffer = ""
        for line in f:
            if begin and start_butch not in line:
                continue
            buffer += line
            if start_butch in line:
                begin = False
                in_batch = True
            if end_butch in line and in_batch:
                in_batch = False
                yield buffer  
                buffer = ""  

        if buffer and in_batch:
            yield buffer


def get(data, level, *, begin='.//PC-Urn_'):
    result = level.find(f'{begin}{data}')
    return result.text if result is not None else None
        
        
def smiles_prepare(smiles):
    print(smiles)
    return smiles

def process(butch: str):
    compress_data = ('cid', 'IUPAC Name', 'SMILES')
    root = ET.fromstring(butch)
    cid = root.find('.//PC-CompoundType_id_cid').text
    props = root.find('.//PC-Compound_props')
    info_data = props.findall('.//PC-InfoData')
    first_values = {'cid': cid}  

    for data in info_data:
        urn = data.find('.//PC-Urn')
        if urn is None:
            continue
        label = get('label', level=urn)
        if label not in compress_data:
            continue
        if label in first_values:
            continue
        
        value = get('.//PC-InfoData_value_sval', level=data, begin='')
        first_values[label] = value
        if len(first_values) == len(compress_data):
            break
        
    utility_df = pd.DataFrame([first_values])
    utility_df['SMILES'] = utility_df['SMILES'].apply(smiles_prepare)
    return utility_df

















def debug():
    xml_file = join(xml_pathdir, 'Compound_000000001_000500000.xml.gz')
    with open('output.csv', mode='w', encoding='utf-8') as f:
        for i, batch in enumerate(read_batches(xml_file), start=1):
            if i == 123:
                break
            utility_df: pd.DataFrame = process(batch)
            df = utility_df.to_csv(sep=';', index=False, header=(i==1), quoting=csv.QUOTE_ALL)
            f.write(df)
            break
    
        
        
        
        


def process_file(xml_file):
    try:
        xml_path = join(xml_pathdir, xml_file)
        csv_file = basename(f"{xml_file.removesuffix('.xml.gz')}.csv")
        csv_path = join(csv_pathdir, csv_file)
        with open(csv_path, mode='w', encoding='utf-8') as f:
            for i, batch in enumerate(read_batches(xml_path), start=1):
                utility_df = process(batch)
                df = utility_df.to_csv(sep=';', index=False, header=(i==1), quoting=csv.QUOTE_ALL)
                f.write(df)
    except Exception as e:
        logger.error(e)


def multymain(load):
    from concurrent.futures import ProcessPoolExecutor
    csv_files = [f for f in os.listdir(xml_pathdir) if f.endswith('.xml.gz')]
    num_workers = max(1, round(os.cpu_count() * (load / 100)))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_file, csv_files)




if __name__ == "__main__":
    load = 70
    debug()
    os._exit(0)




