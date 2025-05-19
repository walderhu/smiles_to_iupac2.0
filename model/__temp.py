import json
from os.path import exists
from typing import *

import numpy as np

from __features import *

dirname = '/home/lipatovdn/__data__'

if not exists(dirname):
    raise FileNotFoundError(f"Нет папки с обучающей выборкой по пути: {dirname}")


def take_dict(filename):
    if exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    raise FileNotFoundError(f"Нет папки с обучающей выборкой по пути: {dirname}")


metaline_dict: dict = take_dict('lineinfo.json')
metadict: dict = take_dict('data-info.json')
batch_size = 256


total_count_lines = sum(metaline_dict.values())
total_steps = (total_count_lines // batch_size)
warmup_steps = 800
base_lr = 0.001
cur_step = sum(metaline_dict[file] for file, prepared in metadict.items() if prepared) // 256

def get_lr(current_step, decay_percentage=0.2, min_lr=1e-4):
    if current_step < warmup_steps:
        return base_lr * (current_step / warmup_steps)
    else:
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        adjusted_progress = progress ** (1 / (1 + decay_percentage))
        lr = min_lr + (base_lr - min_lr) * (1 + np.cos(np.pi * adjusted_progress)) * 0.5
        return lr


steps = range(cur_step, total_steps)
lr_values = [get_lr(step) for step in steps]
send_photo(lr_values)


# import matplotlib.pyplot as plt  
# plt.figure(figsize=(12, 6))  
# plt.plot(steps, lr_values)  
# plt.xlabel("Step")  
# plt.ylabel("Learning Rate")  
# plt.title("Learning Rate Schedule")  
# plt.grid(True)  
# # plt.savefig('learning_rate_schedule.png')  
# plt.show()  