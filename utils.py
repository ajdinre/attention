import torch
import time

from torchinfo import summary
from torchviz import make_dot


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            raise ValueError("Timer is not started")
        elapsed_time = time.time() - self.start_time
        self.start_time = None
        return elapsed_time


def log_model_summary(model, input_size):
    src = torch.randint(0, 1000, input_size).to(next(model.parameters()).device)
    tgt = torch.randint(0, 1000, input_size).to(next(model.parameters()).device)
    print(summary(model, input_data=(src, tgt)))

def visualize_model(model, input_size, filename='model_graph'):
    src = torch.randint(0, 1000, input_size).to(next(model.parameters()).device)
    tgt = torch.randint(0, 1000, input_size).to(next(model.parameters()).device)
    y = model(src, tgt)
    
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render(filename, format='png', cleanup=True)
    print(f"Model graph saved as {filename}.png")