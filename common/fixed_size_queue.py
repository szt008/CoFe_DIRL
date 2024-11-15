from queue import Queue

class FixedSizeQueue:
    def __init__(self, maxsize = 1) -> None:
        self.queue = Queue(maxsize=maxsize)
        pass

    def add(self, item):
        if self.queue.full():
            self.queue.get()
        self.queue.put(item)

    def get(self):
        return self.queue.get()