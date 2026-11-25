"""Explicit forward-only capture helper. Call only with a trusted model and input."""
from .recorder import Recorder


def capture_forward(model, root, identifier=None, *, args=(), kwargs=None, **limits):
    """No backward or optimizer step. Preserves training flags; forward may have side effects."""
    import torch
    with Recorder(model, root, identifier, **limits) as recorder:
        with torch.no_grad():
            model(*args, **(kwargs or {}))
        recorder.snapshot_parameters(gradient_label='forward_only_existing_grad_state')
    return recorder.path
