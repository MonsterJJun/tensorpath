"""Reproducible healthy vs intentionally detached gradient path."""
import argparse
import torch
from torch import nn
from tensorpath import Recorder


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 16)
        self.block = nn.TransformerEncoderLayer(16, 4, 32, dropout=0, batch_first=True)
        self.head = nn.Linear(16, 32)

    def forward(self, x, *, detach=False):
        h = self.block(self.embedding(x))
        return self.head(h.detach() if detach else h)


def run(root, name, detach):
    torch.manual_seed(7)
    model = TinyTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    optimizer.zero_grad(set_to_none=True)
    with Recorder(model, root, name) as trace:
        logits = model(x, detach=detach)
        ce = nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
        auxiliary = logits.square().mean()
        trace.probe_gradients({"ce": ce, "auxiliary": auxiliary}, {"embedding": model.embedding.weight, "head": model.head.weight})
        (ce + auxiliary).backward()
        trace.snapshot_parameters(optimizer, gradient_label="after_backward_before_step_unscaled")
    print(trace.path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    run(args.root, "healthy", False)
    run(args.root, "detached", True)
