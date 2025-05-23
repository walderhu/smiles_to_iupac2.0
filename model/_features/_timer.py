import logging
import signal
import time




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