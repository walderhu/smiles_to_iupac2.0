from csv import QUOTE_ALL
from itertools import islice
from os.path import basename

from pandas import DataFrame

BATCH_SIZE = 500
TOKEN_SIZE = 8


def read_in_batches(file_path):
    with open(file_path, 'r') as f:
        while True:
            batch = list(islice(f, BATCH_SIZE))
            if not batch:
                break
            yield batch


def split(line: str, separator="\t-"):
    if separator in line:
        parts = line.split(separator, 1)
        return parts[0].strip(), parts[1].strip()
    else:
        return line.strip(), None


def parse_vocab(lines):
    return DataFrame([{'token': token, 'index': index_str}
                      for token, index_str in (split(line) for line in lines)
                      if index_str and len(token) <= TOKEN_SIZE])


def handle():
    filepath = "__tokens__/Compound_145500001_146000000.vocab"
    csvfilename = basename(filepath).replace('.vocab', '.csv')
    for num_batch, batch in enumerate(read_in_batches(filepath), start=1):
        with open(csvfilename, mode='w', encoding='utf-8') as f:
            utility_df = parse_vocab(batch)
            df = utility_df.to_csv(sep=';', index=False, header=(num_batch == 1), quoting=QUOTE_ALL)
            f.write(df)


if __name__ == '__main__':
    handle()