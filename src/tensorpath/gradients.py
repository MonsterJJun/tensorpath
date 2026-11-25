"""Explicit gradient observations. No optimizer step or .grad mutation."""
import itertools
import math
import torch


def vector_stats(tensor, max_elements):
    if tensor is None:
        return {"status": "missing", "norm": None}, None
    if tensor.layout != torch.strided or tensor.is_complex() or tensor.device.type == "meta":
        return {"status": "unsupported_layout", "norm": None}, None
    if tensor.numel() > max_elements:
        return {"status": "size_limit", "norm": None}, None
    try:
        v = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        if not bool(torch.isfinite(v).all()):
            return {"status": "nonfinite", "norm": None}, None
        norm = float(torch.linalg.vector_norm(v))
        if not math.isfinite(norm):
            return {"status": "nonfinite", "norm": None}, None
        return {"status": "zero" if norm == 0 else "observed", "norm": norm}, v
    except (RuntimeError, TypeError, NotImplementedError):
        return {"status": "unavailable", "norm": None}, None


def objective_gradients(losses, targets, max_elements=1_000_000):
    """Probe selected scalar losses against explicit tensors; retains existing graph.

    User calls before backward; graph must still be available. Per-objective gradients
    are unweighted unless caller supplies weighted loss. Never writes .grad.
    """
    if not losses or not targets or len(losses) > 8 or len(targets) > 32:
        raise ValueError("Use 1–8 objectives and 1–32 explicit targets")
    for mapping in (losses, targets):
        if any(not isinstance(k, str) or not k or len(k) > 256 or not isinstance(v, torch.Tensor) for k, v in mapping.items()):
            raise ValueError("Expected named tensors")
    if any(loss.ndim != 0 for loss in losses.values()):
        raise ValueError("Objectives must be scalar tensors")
    if len({id(t) for t in targets.values()}) != len(targets):
        raise ValueError("Duplicate target tensors; use one canonical name")
    active = {k: t for k, t in targets.items() if t.requires_grad and t.numel() <= max_elements and t.layout == torch.strided}
    if sum(t.numel() for t in active.values()) * len(losses) > 4_000_000:
        raise ValueError("Gradient probe exceeds total element budget; select fewer targets")
    rows, vectors = [], {}
    for objective, loss in losses.items():
        observed = {}
        failure = None
        if active and loss.requires_grad:
            try:
                observed = dict(zip(active, torch.autograd.grad(loss, tuple(active.values()), retain_graph=True, allow_unused=True, materialize_grads=False)))
            except (RuntimeError, NotImplementedError):
                failure = "autograd_error"
        for name, target in targets.items():
            if not target.requires_grad:
                stats, vector = {"status": "frozen", "norm": None}, None
            elif name not in active:
                stats, vector = {"status": "size_limit" if target.numel() > max_elements else "unsupported_layout", "norm": None}, None
            elif not loss.requires_grad or failure:
                stats, vector = {"status": failure or "loss_no_grad", "norm": None}, None
            else:
                stats, vector = vector_stats(observed.get(name), max_elements)
                if stats["status"] == "missing":
                    stats["status"] = "unused"
            rows.append({"objective": objective, "target": name, **stats})
            vectors[objective, name] = vector
    comparisons = []
    for name in targets:
        for left, right in itertools.combinations(losses, 2):
            a, b = vectors[left, name], vectors[right, name]
            cosine = None
            if a is not None and b is not None:
                na, nb = float(a.norm()), float(b.norm())
                if na and nb:
                    cosine = max(-1.0, min(1.0, float(torch.dot(a / na, b / nb))))
            comparisons.append({"target": name, "left": left, "right": right, "cosine": cosine})
    return rows, comparisons
