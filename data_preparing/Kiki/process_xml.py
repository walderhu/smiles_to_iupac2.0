import pandas as pd
import xml.etree.ElementTree as ET
import gzip, csv, os
from typing import List
from os.path import join, basename
from _notification import send_msg
from _logger import Logger

_debug = True
logger = Logger(debug=_debug, error_filename='error_process_xml.log')
csv_pathdir = '__metadata__'
xml_pathdir = '__xml_data__'
os.makedirs(csv_pathdir, exist_ok=True)
start_butch = '<PC-Compound>'
end_butch = '</PC-Compound>'

def read_batches(xml_file):
    """
    Корутина для взятия батча размером в описание 1 молекулы из архива
    """
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


def process(butch: str):
    _compress_data = True
    compress_data = ('IUPAC Name', 'SMILES', 'Molecular Weight', 'Count')
    root = ET.fromstring(butch)
    props = root.find('.//PC-Compound_props')
    info_data = props.findall('.//PC-InfoData')

    utility_data = []
    for data in info_data:
        urn = data.find('.//PC-Urn')
        if urn is None:
            continue
        
        def get(data, *, begin='.//PC-Urn_', level=urn):
            result = level.find(f'{begin}{data}')
            return result.text if result is not None else None

        label = get('label')
        if _compress_data and label not in compress_data:
            continue
        
        utility_data.append({
            'cid' : root.find('.//PC-CompoundType_id_cid').text,
            'label': label,
            'name': get('name'),
            'implementation' : get('implementation'),
            'version' : get('version'),
            'software' : get('software'),
            'source' : get('source'),
            'release': get('release'),
            'value' : get('.//PC-InfoData_value_sval', begin='', level=data)
            })

    utility_df = pd.DataFrame(utility_data)
    return utility_df
    

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
    from concurrent.futures import ProcessPoolExecutor, as_completed
    csv_files = [f for f in os.listdir(xml_pathdir) if f.endswith('.xml.gz')]
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




def debug():
    """
    открываем первый архив, в котором берем батч под номером 123
    и далее обрабатываем его через процесс
    """
    xml_file = join(xml_pathdir, 'Compound_000000001_000500000.xml.gz')
    print(xml_file)
    b: str
    for i, batch in enumerate(read_batches(xml_file), start=1):
        if i == 123:
            b = batch
            break
        
    utility_df = process(b)
    utility_df.to_csv('output.csv', sep=';', index=False, \
        encoding='utf-8', quoting=csv.QUOTE_ALL, mode='w', header=True)
    return 


if __name__ == "__main__":
    load = 70
    # count = multymain(load)
    # send_msg(f"Successfully processed {count} files")
    debug()
    os._exit(0)




# fields = {
#     'cid': lambda: root.find('.//PC-CompoundType_id_cid').text,
#     'label': lambda: label,
#     'name': lambda: get('name'),
#     'implementation': lambda: get('implementation'),
#     'version': lambda: get('version'),
#     'software': lambda: get('software'),
#     'source': lambda: get('source'),
#     'release': lambda: get('release'),
#     'value': lambda: get('.//PC-InfoData_value_sval', begin='', level=data)
# }

# utility_data.append({key: func() for key, func in fields.items()})
