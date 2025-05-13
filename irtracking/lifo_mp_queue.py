from multiprocessing import Manager, Lock
from queue import Empty

class LifoMPQueue:
    def __init__(self, maxsize=32):
        self.manager = Manager()
        self.deque = self.manager.list()
        self.lock = Lock()
        self.maxsize = maxsize

    def put(self, item):
        with self.lock:
            if len(self.deque) >= self.maxsize:
                self.deque.pop(0)  # Remove oldest (leftmost)
            self.deque.append(item)

    def get(self, block=True, timeout=None):
        import time
        start = time.time()
        while True:
            with self.lock:
                if self.deque:
                    return self.deque.pop()  # Get newest (rightmost)
            if not block:
                raise Empty
            if timeout is not None and (time.time() - start) > timeout:
                raise Empty
            time.sleep(0.001)

    def qsize(self):
        with self.lock:
            return len(self.deque) 