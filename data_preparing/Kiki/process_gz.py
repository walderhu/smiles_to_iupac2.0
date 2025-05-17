from io import StringIO, BytesIO
from chython import SDFRead
from datetime import datetime
from typing import *
from rdkit import Chem
import csv
import gzip
import os
import logging
import rdkit.Chem.MolStandardize.rdMolStandardize as rdkit
import pkg_resources

"""
Настройки
"""
_debug = False
_format= '%(message)s [File: \"%(filename)s\", line %(lineno)d]' if _debug else '%(message)s'
logging.basicConfig(level=logging.INFO, format=_format)

version_chython = pkg_resources.get_distribution("chython").version
version_rdkit = pkg_resources.get_distribution("rdkit").version
gz_pathdir = '__archive_data__'
csv_pathdir = '__gz_to_csv_data__'
os.makedirs(csv_pathdir, exist_ok=True)

datetime_format: str = "%Y-%m-%dT%H:%M:%S"
rows: List[str] = [
    "PUBCHEM_COMPOUND_CID",
    "PUBCHEM_SMILES",
    "CHYTHON_SMILES",
    "RDKIT_SMILES",
    "PUBCHEM_IUPAC_NAME",
    "STANDARDIZE_CHYTHON",
    "STANDARDIZE_RDKIT",
    "PUBCHEM_IUPAC_OPENEYE_NAME",
    "PUBCHEM_IUPAC_CAS_NAME",
    "PUBCHEM_IUPAC_NAME_MARKUP",
    "PUBCHEM_IUPAC_SYSTEMATIC_NAME",
    "PUBCHEM_IUPAC_TRADITIONAL_NAME",
    "PUBCHEM_OPENEYE_CAN_SMILES",
    "PUBCHEM_OPENEYE_ISO_SMILES",
    "DATATIME",
    "VERSION_CHYTHON",
    "VERSION_RDKIT",
    "ARCHIVE_NAME"]
delimiter_pubchem: str = "$$$$"


"""
Вспомогательная функция для итерирования по архиву

———
Мы итерируемся по батчу, равному информации об 1 молекуле
"""
def get_molecule_data(archive_iterator: Iterator[str]) -> Iterator[str]:
    for line in archive_iterator:
        yield line
        if delimiter_pubchem in line:
            next(archive_iterator)
            break


"""
Парсит скачанный архив и достает оттуда нужную информацию
"""
def parsing_pubchem_gzip(arch_name: str):    
    arch_path = os.path.join(gz_pathdir, arch_name)
    with gzip.open(arch_path, encoding='utf-8', mode='rt') as arch:
        archive_iterator: Iterator = iter(arch)
        csv_name = os.path.basename(f"{arch_name.removesuffix('.sdf.gz')}.csv")
        csv_path = os.path.join(csv_pathdir, csv_name)
        
        with open(csv_path, encoding='utf-8', mode='w', newline='') as file:
            file_writer = csv.DictWriter(file, delimiter=';', fieldnames=rows)
            file_writer.writeheader()
            counter = 0
            while _lines := get_molecule_data(archive_iterator):
                sdf = ''.join(line for line in _lines)
                data: Dict[str, Optional[str]] = dict.fromkeys(rows, None)
                
                lines: List[str] = sdf.split('\n')
                for i, line in enumerate(lines, start=1):
                    for key in data.keys():
                        if key in line:
                            data[key] = lines[i].rstrip("\n")

                if smiles := data.get("PUBCHEM_SMILES"):
                    if counter != 0:
                        sdf = f'\n{sdf}'
                        
                    try:
                        with StringIO(sdf) as f, SDFRead(f) as r:
                            smiles_chython = next(r)
                            
                        smiles_chython.standardize(logging=True, ignore=False,
                            fix_tautomers=True, _fix_stereo=True)
                        standardize_chython = str(smiles_chython)
                        
                        with BytesIO(sdf.encode('utf-8')) as sdf_file:
                            smiles_rdkit = next((Chem.MolToSmiles(mol) \
                                for mol in Chem.ForwardSDMolSupplier(sdf_file) if mol))
                        standardize_rdkit = rdkit.StandardizeSmiles(smiles_rdkit)
                        
                        date = datetime.now().strftime(datetime_format)
                        data.update({
                            "CHYTHON_SMILES" : smiles_chython, "RDKIT_SMILES" : smiles_rdkit,
                            "STANDARDIZE_CHYTHON": standardize_chython, "STANDARDIZE_RDKIT": standardize_rdkit,
                            "DATATIME": date, "VERSION_CHYTHON" : version_chython,
                            "VERSION_RDKIT" : version_rdkit, "ARCHIVE_NAME" : arch_name})
                    except Exception as e:
                        logging.error(f"Произошло исключение с {data.get('PUBCHEM_COMPOUND_CID')} {smiles} {e}")
                file_writer.writerow(data)
                
                counter += 1


def wrapper(archs):
    try:
        parsing_pubchem_gzip(archs)
    except Exception as e:
        pass

def main():
    archs: list = os.listdir(gz_pathdir)
    for arch in archs:
        parsing_pubchem_gzip(arch)

def multymain():
    from concurrent.futures import ProcessPoolExecutor
    archs = os.listdir(gz_pathdir)
    print(archs)
    num_workers = round(os.cpu_count() * 0.5)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(wrapper, archs)
    
    
if __name__ == '__main__':
    # multymain()
    os._exit(0)
