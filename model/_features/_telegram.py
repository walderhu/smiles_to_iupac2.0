import shlex
import subprocess

import lithium




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


def send_photo(*args, **kwargs):
    return lithium.send_photo(*args, **kwargs)


def send_msg(*args, **kwargs):
    return lithium.send_msg(*args, **kwargs)
