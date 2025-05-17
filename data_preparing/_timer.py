import time
import signal
import logging


logging.basicConfig(level=logging.DEBUG, format='%(name)s: %(message)s')  

class Timer:
    def __init__(self, name="Timer", *, max_time=None):
        self.start_time = None
        self.max_time = max_time
        self.timed_out = False
        self.TimeoutException = self.TimeoutException
        self.logger = logging.getLogger(name)

    def _signal_handler(self, signum, frame):
        self.timed_out = True
        raise self.TimeoutException(f"Время выполнения превысило {self.max_time} секунд.")

    def __enter__(self):
        self.start_time = time.time()
        if self.max_time:
            signal.signal(signal.SIGALRM, self._signal_handler)
            signal.alarm(self.max_time)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
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
            self.logger.debug(f"Время выполнения: {elapsed_time:.4f} сек")  
            return False

    class TimeoutException(Exception):
        pass


if __name__ == '__main__':
    with Timer(name="MyTask", max_time=2):
        try:
            time.sleep(3)  
        except Exception as e:
            print(f"Обработано исключение внутри блока: {e}")

    print("Программа продолжает выполнение.")