"""
Модуль логгирования с классом Logger, обеспечивающим гибкую настройку логирования.

Класс Logger конфигурирует логгер с тремя обработчиками:
- Консольный обработчик (StreamHandler), выводящий сообщения уровней DEBUG, INFO и WARNING.
- Файловый обработчик для общих логов (если указан файл), принимающий сообщения уровней DEBUG, INFO и WARNING.
- Файловый обработчик для ошибок (error.log), записывающий сообщения уровня ERROR и выше.

Фильтры LevelFilter и MaxLevelFilter используются для ограничения уровней сообщений, которые обрабатывает каждый обработчик.

Класс Logger также реализует метод __getattr__, позволяющий напрямую вызывать методы стандартного логгера.

Автор: Денис Липатов 
Дата: 2025
"""

import logging
import sys


class Logger:
    def __init__(self, *, debug=False, log_file='', error_filename='error.log', namelog=__name__):
        self._logger = logging.getLogger(namelog)
        self._logger.setLevel(logging.DEBUG)

        if log_file:
            file_format = '%(message)s [File: "%(filename)s", line %(lineno)d]' \
                if debug else '%(message)s'
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(file_format))
            file_handler.addFilter(self.MaxLevelFilter(max_level=logging.ERROR))
            self._logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        console_handler.addFilter(self.MaxLevelFilter(max_level=logging.ERROR))
        self._logger.addHandler(console_handler)

        error_handler = logging.FileHandler(error_filename)
        error_handler.setLevel(logging.ERROR)
        error_format = 'Exception: %(message)s [File: "%(filename)s", line %(lineno)d]'
        error_handler.setFormatter(logging.Formatter(error_format))
        error_handler.addFilter(self.LevelFilter(min_level=logging.ERROR))
        self._logger.addHandler(error_handler)


    def __getattr__(self, item):
        return getattr(self._logger, item)
    
    
    class LevelFilter(logging.Filter):
        def __init__(self, min_level=logging.INFO):
            super().__init__()
            self.min_level = min_level

        def filter(self, record):
            return record.levelno >= self.min_level

    class MaxLevelFilter(logging.Filter):
        def __init__(self, max_level):
            super().__init__()
            self.max_level = max_level

        def filter(self, record):
            return record.levelno < self.max_level


def main():
    logger = Logger()
    try:
        1 / 0
    except Exception as e:
        logger.error("Caught an exception", exc_info=True)

if __name__ == '__main__':
    main()
    
__all__ = ['Logger']
