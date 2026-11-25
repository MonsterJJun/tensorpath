"""Architecture-agnostic forward-only traces; no training or optimizer."""
import argparse
import torch
from torch import nn
from tensorpath import Recorder


class Branched(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(8, 4)
        self.right = nn.Linear(8, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.left(x) + self.right(x))


def main(root):
    torch.manual_seed(7)
    cases = [
        ('cnn_forward', nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4)), torch.randn(2, 3, 16, 16)),
        ('lstm_forward', nn.LSTM(8, 16, batch_first=True), torch.randn(2, 5, 8)),
        ('branched_forward', Branched(), torch.randn(2, 8)),
    ]
    for name, model, x in cases:
        model.eval()
        with Recorder(model, root, name) as trace:
            with torch.no_grad():
                model(x)
        print(trace.path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    main(parser.parse_args().root)
