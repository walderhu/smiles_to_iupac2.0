import itertools
import csv
from os.path import basename
from pandas import DataFrame



def split(line: str, separator = "\t-"):
    if separator in line:
        parts = line.split(separator, 1)
        return parts[0].strip(), parts[1].strip()
    else:
        return line.strip(), None


def parse_vocab(lines):
    return DataFrame([{'token': token, 'index' : index_str} \
        for token, index_str in (split(line) for line in lines)
        if index_str])


def read_in_batches(file_path, batch_size=500):
    with open(file_path, 'r') as f:
        while True:
            batch = list(itertools.islice(f, batch_size))
            if not batch:
                break
            yield batch


filepath = "__tokens__/Compound_145500001_146000000.vocab"
csvfilename = basename(filepath).replace('.vocab', '.csv')
for num_batch, batch in enumerate(read_in_batches(filepath), start=1):
    with open(csvfilename, mode='w', encoding='utf-8') as f:
        utility_df = parse_vocab(batch)
        df = utility_df.to_csv(sep=';', index=False, header=(num_batch == 1), quoting=csv.QUOTE_ALL)
        f.write(df)
