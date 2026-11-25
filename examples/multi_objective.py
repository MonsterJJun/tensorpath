"""Selected soft embedding target: KL + CE, with frozen model weights."""
import argparse
import torch
from torch import nn
from torch.nn import functional as F
from tensorpath import Recorder


def main(root):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 7))
    model.requires_grad_(False)
    embedding = nn.Parameter(torch.randn(1, 16))
    optimizer = torch.optim.AdamW([embedding])
    teacher = torch.randn(1, 7)
    tau = 2.0
    with Recorder(model, root, "soft_embedding") as trace:
        logits = model(embedding)
        ce = F.cross_entropy(logits, torch.tensor([2]))
        kd = F.kl_div(F.log_softmax(logits / tau, -1), F.softmax(teacher / tau, -1), reduction="batchmean") * tau**2
        trace.probe_gradients({"ce": ce, "kd": kd}, {"soft_embedding": embedding})
        ((1-.9)*ce + .9*kd).backward()
        trace.snapshot_parameters(optimizer, gradient_label="after_backward_before_step_unscaled", extra_targets={"soft_embedding": embedding})
    print(trace.path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    main(parser.parse_args().root)
