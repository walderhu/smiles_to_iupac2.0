import traceback

import torch



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
