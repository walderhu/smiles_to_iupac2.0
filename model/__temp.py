import json
import logging
import os
from os.path import basename, exists, join

from _model import batch_reader

png_dirname = '__media__'
os.makedirs(png_dirname, exist_ok=True)

data_dirname = '/home/lipatovdn/__data__'
log_filename = 'klasterisation.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')


meta = 'data-info.json'
if exists(meta):
    with open(meta, 'r') as f:
        files_dict = json.load(f)
else:
    files = [join(data_dirname, f) for f in os.listdir(data_dirname)
             if f.endswith('.csv') and not f.startswith('_')]
    files_dict = {file: False for file in files}


def medianame(file, num_batch):
    base: str = basename(file)
    res = base.replace('.csv', '.png')
    res = f"{'clusters_'}{num_batch}_of_{res}"
    res = join(png_dirname, res)
    return res


files = [file for f in os.listdir(data_dirname)
         if f.endswith('.csv') and
         not f.startswith('_') and
         files_dict[file := join(data_dirname, f)] is False][:3]

valid_repr = []
for file in files:
    for num_batch, batch in enumerate(batch_reader(join(data_dirname, file), batch_size=2e5), start=1):
        pngfilename = medianame(file, num_batch)
        files_dict[file] = True

with open(meta, 'w') as f:
    json.dump(files_dict, f)
