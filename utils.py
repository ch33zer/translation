from time import perf_counter
import sys
import torch

def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class runtime:
    def __enter__(self):
        self.start = perf_counter()
        return self
    def curr(self):
        return perf_counter() - self.start
    def __exit__(self, type, value, traceback):
        pass
