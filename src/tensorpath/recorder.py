"""Opt-in eager-mode recorder. Stores metadata, never tensor values."""
from collections import Counter
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import platform
import time
import uuid

import torch
from .gradients import objective_gradients, vector_stats
from .store import save_trace, trace_id, MAX_ROWS


def tensor_metadata(t, path):
    try:
        shape = list(t.shape)
        count = t.numel()
        return {"path": path[:512], "shape": shape, "dtype": str(t.dtype), "device": str(t.device),
                "layout": str(t.layout), "requires_grad": t.requires_grad, "numel": count,
                "logical_bytes": count * t.element_size()}
    except (RuntimeError, ValueError, TypeError, NotImplementedError):
        return {"path": path[:512], "status": "metadata_unavailable"}


def describe(value, limit):
    rows, seen = [], set()
    truncated = False
    visited = 0

    def walk(v, path, depth):
        nonlocal truncated, visited
        visited += 1
        if len(rows) >= limit or depth > 12 or visited > 2048:
            truncated = True
            return
        if isinstance(v, torch.Tensor):
            rows.append(tensor_metadata(v, path))
            return
        if id(v) in seen:
            truncated = True
            return
        if isinstance(v, (Mapping, tuple, list)) or (is_dataclass(v) and not isinstance(v, type)):
            seen.add(id(v))
            try:
                if isinstance(v, Mapping):
                    # Mapping key values are not recorded (could contain private data).
                    items = enumerate(v.values())
                elif isinstance(v, (tuple, list)):
                    items = enumerate(v)
                else:
                    items = ((f.name, getattr(v, f.name)) for f in fields(v))
                for key, child in items:
                    if len(rows) >= limit or visited > 2048:
                        truncated = True
                        break
                    walk(child, f"{path}.{key}", depth + 1)
            finally:
                seen.remove(id(v))
    walk(value, "$", 0)
    return rows, truncated


class Recorder:
    """One context per trace, single-threaded eager execution, explicit snapshots."""
    def __init__(self, model, root, identifier=None, *, max_events=1000, max_tensors=16, max_grad_elements=1_000_000):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be torch.nn.Module")
        for value, upper in ((max_events, 5000), (max_tensors, 64), (max_grad_elements, 1_000_000)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"Limit must be in [1, {upper}]")
        self.model, self.root = model, root
        self.max_events, self.max_tensors, self.max_grad_elements = max_events, max_tensors, max_grad_elements
        self.identifier = trace_id(identifier or uuid.uuid4().hex)
        self.data = {"schema_version": 1, "trace_id": self.identifier, "status": "complete", "metadata": {
            "torch": str(torch.__version__), "python": platform.python_version(), "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_type": type(model).__name__, "training_at_start": model.training, "observation": "eager_module_hooks",
            "parameter_snapshot": "not_captured", "event_limit": max_events, "tensor_limit": max_tensors},
            **{key: [] for key in ("modules", "events", "parameters", "gradients", "comparisons", "warnings")}}
        self._handles, self._stack, self._counts = [], [], Counter()
        self._warning_codes = set()
        self._active = self._used = False
        self.path = None

    def _warn(self, code):
        if code not in self._warning_codes:
            self._warning_codes.add(code)
            self.data["warnings"].append({"code": code})

    def _pre(self, name, args, kwargs):
        self._counts[name] += 1
        if len(self.data["events"]) >= self.max_events:
            self._warn("event_limit_reached")
            self._stack.append((name, None))
            return
        inputs, truncated = describe((args, kwargs), self.max_tensors)
        if truncated:
            self._warn("tensor_metadata_truncated")
        parent = next((e["id"] for _, e in reversed(self._stack) if e is not None), None)
        event = {"id": len(self.data["events"]) + 1, "parent_id": parent, "module": name,
                 "occurrence": self._counts[name], "inputs": inputs, "outputs": [], "status": "started",
                 "_start": time.perf_counter_ns()}
        self.data["events"].append(event)
        self._stack.append((name, event))

    def _post(self, name, output):
        if not self._stack:
            self._warn("unmatched_hook")
            return
        while self._stack and self._stack[-1][0] != name:
            self._stack.pop()
            self._warn("unfinished_module_call")
        if not self._stack:
            self._warn("unmatched_hook")
            return
        _, event = self._stack.pop()
        if event is None:
            return
        event["host_elapsed_ms"] = (time.perf_counter_ns() - event.pop("_start")) / 1e6
        event["outputs"], truncated = describe(output, self.max_tensors)
        event["status"] = "returned"  # no always_call: failures remain unfinished
        if truncated:
            self._warn("tensor_metadata_truncated")

    def __enter__(self):
        if self._used:
            raise RuntimeError("Recorder is single-use")
        self._used = self._active = True
        try:
            modules = list(self.model.named_modules())
            if len(modules) > MAX_ROWS:
                raise ValueError("Too many modules for one trace")
            # Shared modules get one hook and a canonical name, with aliases recorded.
            aliases = {}
            for n, m in self.model.named_modules(remove_duplicate=False):
                aliases.setdefault(id(m), []).append(n or "<root>")
            for n, m in modules:
                name = n or "<root>"
                self.data["modules"].append({"name": name, "type": type(m).__name__, "aliases": aliases[id(m)], "training": m.training})
                self._handles.append(m.register_forward_pre_hook(lambda mod, a, kw, name=name: self._pre(name, a, kw), with_kwargs=True))
                self._handles.append(m.register_forward_hook(lambda mod, a, kw, out, name=name: self._post(name, out), with_kwargs=True))
            return self
        except BaseException:
            self._cleanup()
            raise

    def _cleanup(self):
        self._active = False
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _require_active(self):
        if not self._active:
            raise RuntimeError("Recorder must be used inside its context")

    def snapshot_parameters(self, optimizer=None, *, gradient_label="user_snapshot", extra_targets=None):
        """Call after backward, before zero_grad; repeated call replaces snapshot."""
        self._require_active()
        if not isinstance(gradient_label, str) or not 1 <= len(gradient_label) <= 256:
            raise ValueError("Invalid gradient label")
        if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be torch.optim.Optimizer")
        groups = {} if optimizer is not None else None
        if groups is not None:
            for i, group in enumerate(optimizer.param_groups):
                for p in group["params"]:
                    groups.setdefault(id(p), []).append(i)
        alias = {}
        for name, p in self.model.named_parameters(remove_duplicate=False):
            alias.setdefault(id(p), []).append(name)
        named = list(self.model.named_parameters())
        extra_targets = {} if extra_targets is None else extra_targets
        if not isinstance(extra_targets, Mapping) or len(extra_targets) > 32:
            raise ValueError("extra_targets must contain at most 32 named leaf tensors")
        names = {name for name, _ in named}
        for name, p in extra_targets.items():
            if not isinstance(name, str) or not name or len(name) > 256 or name in names or not isinstance(p, torch.Tensor) or not p.is_leaf or id(p) in alias:
                raise ValueError("extra_targets must be uniquely named leaf tensors outside model parameters")
            alias[id(p)] = [name]
            names.add(name)
            named.append((name, p))
        rows = []
        for name, p in named:
            if len(rows) >= MAX_ROWS:
                self._warn("parameter_limit_reached")
                break
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                meta, status = {"requires_grad": p.requires_grad}, {"status": "uninitialized", "norm": None}
            else:
                meta = tensor_metadata(p, name)
                meta["requires_grad"] = p.requires_grad
                status, _ = vector_stats(p.grad, self.max_grad_elements)
            rows.append({"name": name, "aliases": alias[id(p)], **meta, "grad_status": status["status"], "grad_norm": status["norm"],
                         "optimizer_groups": None if groups is None else groups.get(id(p), [])})
        self.data["parameters"] = rows
        self.data["metadata"]["parameter_snapshot"] = gradient_label
        self.data["metadata"]["optimizer_checked"] = optimizer is not None

    def probe_gradients(self, losses, targets):
        """Probe explicit loss/target tensors before backward. Replaces previous probe."""
        self._require_active()
        rows, comparisons = objective_gradients(losses, targets, self.max_grad_elements)
        self.data["gradients"], self.data["comparisons"] = rows, comparisons

    def __exit__(self, typ, value, tb):
        self._cleanup()
        for event in self.data["events"]:
            if event["status"] == "started":
                event.pop("_start", None)
                event["status"] = "unfinished"
                self._warn("unfinished_module_call")
        self._stack.clear()
        if typ is not None:
            self.data["status"] = "failed"
            self.data["metadata"]["exception_type"] = typ.__name__
        try:
            self.path = save_trace(self.root, self.data)
        except Exception as exc:
            if typ is None:
                raise
            if hasattr(value, "add_note"):
                value.add_note(f"TensorPath could not save failed trace: {type(exc).__name__}")
        return False
