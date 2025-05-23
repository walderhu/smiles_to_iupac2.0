import subprocess
from io import StringIO

import pandas as pd

from _batch_reader import batch_reader



def max_len_file(file) -> int:
    max_length = 0
    for batch in batch_reader(file, batch_size=256):
        df = pd.read_csv(StringIO(batch), delimiter=';', encoding='utf-8').dropna()
        max_length_batch = df['IUPAC Name'].astype(str).str.len().max()
        if max_length_batch > max_length:
            max_length = max_length_batch
    return max_length


def count_lines(filepath):
    command = ['wc', '-l', filepath]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    output = result.stdout.strip()
    line_count = int(output.split()[0])
    return line_count


def cpu_usage():
    top_command = ['top', '-bn1']
    grep_command = ['grep', 'Cpu(s)']
    awk_command = ['awk', '{print 100 - $8"%" }']
    top_process = subprocess.Popen(top_command, stdout=subprocess.PIPE)
    grep_process = subprocess.Popen(grep_command, stdin=top_process.stdout, stdout=subprocess.PIPE)
    awk_process = subprocess.Popen(awk_command, stdin=grep_process.stdout, stdout=subprocess.PIPE, text=True)
    top_process.stdout.close()
    grep_process.stdout.close()
    output, _ = awk_process.communicate()
    cpu_usage = float(output.strip('%\n'))
    return cpu_usage


def cpu_usage():
    """Возвращает число загрузки CPU в процентах."""
    try:
        top_command = ['top', '-bn1']
        grep_command = ['grep', 'Cpu(s)']
        awk_command = ['awk', '{print 100 - $8}']  # Убрал '%' из awk, чтобы возвращалось число
        top_process = subprocess.Popen(top_command, stdout=subprocess.PIPE)
        grep_process = subprocess.Popen(grep_command, stdin=top_process.stdout, stdout=subprocess.PIPE)
        awk_process = subprocess.Popen(awk_command, stdin=grep_process.stdout, stdout=subprocess.PIPE, text=True)
        top_process.stdout.close()
        grep_process.stdout.close()
        output, _ = awk_process.communicate()
        cpu_usage = float(output.strip())  # Удалил '%', т.к. его больше нет в выводе awk
        return cpu_usage
    except FileNotFoundError:
        print("Warning: top command not found. CPU usage monitoring disabled.")
        return 0.0  # Или другое значение по умолчанию
    except Exception as e:
        print(f"Error getting CPU usage: {e}")
        return 0.0


def ram_usage():
    """Возвращает число загрузки RAM в процентах."""
    try:
        free_command = ['free', '-m']
        awk_command = ['awk', '/Mem:/ {print $2, $3}']
        free_process = subprocess.Popen(free_command, stdout=subprocess.PIPE)
        awk_process = subprocess.Popen(awk_command, stdin=free_process.stdout, stdout=subprocess.PIPE, text=True)
        free_process.stdout.close()
        output, _ = awk_process.communicate()
        total_ram, used_ram = map(int, output.strip().split())
        ram_usage = (used_ram / total_ram) * 100
        return ram_usage
    except FileNotFoundError:
        print("Warning: free command not found. RAM usage monitoring disabled.")
        return 0.0
    except Exception as e:
        print(f"Error getting RAM usage: {e}")
        return 0.0


def gpu_usage():
    """Возвращает число загрузки GPU в процентах."""
    try:
        nvidia_smi_command = ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']
        nvidia_smi_process = subprocess.Popen(nvidia_smi_command, stdout=subprocess.PIPE, text=True)
        output, _ = nvidia_smi_process.communicate()
        gpu_usage = float(output.strip())
        return gpu_usage
    except FileNotFoundError:
        print("Warning: nvidia-smi command not found. GPU usage monitoring disabled.")
        return 0.0
    except Exception as e:
        print(f"Error getting GPU usage: {e}")
        return 0.0


def cpu(): return f'CPU {cpu_usage():05.2f}%'
def ram(): return f'RAM {ram_usage():05.2f}%'
def gpu(): return f'GPU {gpu_usage():05.2f}%'


def meta():
    return f"{cpu()}, {ram()}, {gpu()}"