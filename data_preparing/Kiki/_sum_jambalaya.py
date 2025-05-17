from collections import Counter
from functools import reduce
import pandas as pd
import operator
import pickle

from pprint import pprint
from os.path import join
import os

def sum(*counters):
    return reduce(operator.add, counters)

dirname = join(os.getcwd(), '__count__')
files = [join(dirname, f) for f in os.listdir(dirname) if f.endswith('.pkl')]

counters = []
for file in files:
    with open(file, "rb") as f:
        deserialized_data = pickle.load(f)
        counters.append(deserialized_data)

total: Counter = sum(*counters)


def print_total_info():
    df = pd.DataFrame(total.items(), columns=['Process', 'Count'])
    df = df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    print(df.to_string(index=False))
    print(f'Всего на данный момент {len(counters)} файла')

def print_percent_info():
    total_dict = dict(total)
    for key, value in total_dict.items():
        res = (value / total_dict['All']) * 100
        print(f'{key}: {res:.2f}%')


print_percent_info()