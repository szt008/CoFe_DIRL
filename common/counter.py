

class Counter:
    def __init__(self, start=1, gap=1) -> None:
        self.data = start
        self.gap = gap
    def count(self):
        self.data += self.gap
        return self.data - self.gap
    def value(self):
        return self.data - self.gap