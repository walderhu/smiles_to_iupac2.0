import logging
import shlex
import signal
import subprocess
import time
import traceback
from io import StringIO

import lithium
import pandas as pd
import torch

from _model import batch_reader


def log_err_msg_wrap(exc, min_d=85, max_d=105) -> str:
    """
    Оборачивает сообщение об ошибке, гарантируя минимальную и максимальную ширину строки.
    """
    message = str(exc)
    wrapped_message = ""
    current_line = ""
    words = message.split()
    for word in words:
        word_len = len(word)
        if len(current_line) == 0:
            if word_len > max_d:
                wrapped_message += current_line.rstrip() + "\n"
                wrapped_message += word + "\n"
                current_line = ""
            else:
                current_line += word + " "
        elif len(current_line) + word_len + 1 <= max_d:
            current_line += word + " "
        else:
            if len(current_line) >= min_d:
                wrapped_message += current_line.rstrip() + "\n"
                current_line = word + " "
            else:
                wrapped_message += current_line.rstrip() + "\n"
                wrapped_message += word + " "
                current_line = ""
    wrapped_message += current_line.rstrip()
    return wrapped_message.rstrip()


def get_clickable_traceback(exc):
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    clickable_lines = []
    for line in tb_lines:
        if "File \"" in line:
            parts = line.split(", line ", 1)
            if len(parts) > 1:
                file_path = parts[0].split("File \"")[1].strip().replace("\\", "/")
                line_number = parts[1].split(", in")[0].strip()
                clickable_lines.append(f"{file_path}:{line_number}: {line.strip()}")
            else:
                clickable_lines.append(line.strip())
        else:
            clickable_lines.append(line.strip())
    return "\n".join(clickable_lines)


# def err_wrap(basename, exc):
#     clickable_tb = get_clickable_traceback(exc)
#     exc_type = type(exc).__name__
#     exc_message_content = str(exc) # The actual error message
#     exc_message = f"""\n\n\n⚠️  CRITICAL ERROR in {basename} ⚠️

# {log_err_msg_wrap(clickable_tb)}

# Final Error Message: {exc_type}: {exc_message_content}

# 🖥️  GPU Memory Info:
# Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB
# Reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB
#     """
#     return exc_message


def err_wrap(basename: str, exc: Exception) -> str:
    """Форматирует сообщение об ошибке в понятном виде."""
    # Получаем тип ошибки и сообщение
    error_type = type(exc).__name__
    error_msg = str(exc)

    # Формируем трассировку стека
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)

    # Обрабатываем трассировку для лучшей читаемости
    simplified_traceback = []
    for line in tb_lines:
        # Упрощаем пути к файлам и номера строк
        if "File \"" in line and ", line " in line:
            file_part, rest = line.split(", line ", 1)
            file_path = file_part.split("File \"")[1].strip()
            line_num = rest.split(",")[0]
            func_name = rest.split(", in ")[1].strip() if ", in " in rest else ""
            simplified_traceback.append(f"→ {file_path}:{line_num} (in {func_name})")
        else:
            simplified_traceback.append(line.strip())

    # Формируем финальное сообщение
    message = f"""
⚠️ CRITICAL ERROR in {basename} ⚠️

{'-'*60}
ERROR TYPE: {error_type}
MESSAGE: {error_msg}
{'-'*60}

TRACEBACK (most recent call last):
{"-"*60}
""" + "\n".join(simplified_traceback[-10:]) + f"""
{"-"*60}

GPU MEMORY INFO:
→ Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB
→ Reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB
"""
    return message


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


# def msg(message):
#     command = ['msg', message]
#     result = subprocess.run(command, capture_output=True, text=True, check=True)
#     output = result.stdout.strip()
#     return output


def msg(message):
    try:
        command = ['msg'] + shlex.split(message)
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        return output
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении команды 'msg': {e}")
        print(f"stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        print("Команда 'msg' не найдена. Убедитесь, что она установлена и доступна в PATH.")
        raise
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        raise


class Timer:
    def __init__(self, name:str="Timer", *, max_time:int=None, logger:logging.Logger=None):
        self.start_time = None
        self.max_time = max_time
        self.timed_out = False
        self.TimeoutException = self.TimeoutException
        if logger is None:
            print('Логгер не передан')
            raise ValueError("Logger not found or not provided.  A logger instance must be passed to the constructor.")
        else:
            self.logger = logger

    def _signal_handler(self, signum, frame):
        self.timed_out = True
        raise self.TimeoutException(f"Время выполнения превысило {self.max_time} секунд.")

    def __enter__(self):
        self.start_time = time.time()
        if self.max_time:
            signal.signal(signal.SIGALRM, self._signal_handler)
            signal.alarm(self.max_time)
        return self

    @property
    def status(self):
        if not self.start_time:
            return 0
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        return round(elapsed_time, 4)

    def __float__(self):
        return self.status

    def __repr__(self):
        return f'{self.status:05.2f}'

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_time - self.start_time
        if self.max_time:
            signal.alarm(0)

        if exc_type is self.TimeoutException:
            self.logger.error(f"Таймаут! {exc_val}")
            return True
        elif exc_type:
            import traceback
            traceback.print_tb(exc_tb)
            self.logger.error(f"Произошла ошибка: {exc_val}")
            return False
        elif self.timed_out:
            self.logger.warning("Таймаут произошел, но исключение уже было обработано.")
            return True
        else:
            # self.logger.debug(f"Время выполнения: {elapsed_time:.4f} сек")
            return False

    class TimeoutException(Exception):
        pass


# def count_lines(filepath):
#     print(f'counting for {basename(filepath)}')
#     command = ['wc', '-l', filepath]
#     result = subprocess.run(command, capture_output=True, text=True, check=True)
#     output = result.stdout.strip()
#     line_count = int(output.split()[0])
#     return int(line_count)


def send_photo(*args, **kwargs):
    return lithium.send_photo(*args, **kwargs)


def send_msg(*args, **kwargs):
    return lithium.send_msg(*args, **kwargs)
