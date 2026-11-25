"""Versioned, bounded, JSON-only trace storage."""
import json
import os
import re
import stat
import tempfile
from pathlib import Path

MAX_BYTES = 16 * 1024 * 1024
MAX_ROWS = 20_000
ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}\Z")
SECTIONS = ("modules", "events", "parameters", "gradients", "comparisons", "warnings")


class TraceError(ValueError):
    pass


def trace_id(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise TraceError("Invalid trace ID; use 1–80 ASCII letters, digits, underscores or hyphens.")
    return value


def encode(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _check(value, depth=0):
    if depth > 24:
        raise TraceError("Trace nesting exceeds limit")
    if isinstance(value, dict):
        for k, v in value.items():
            if len(k) > 1024:
                raise TraceError("Oversized key")
            _check(v, depth + 1)
    elif isinstance(value, list):
        for v in value:
            _check(v, depth + 1)
    elif isinstance(value, str) and len(value) > 4096:
        raise TraceError("Oversized string")
    elif isinstance(value, float):
        import math
        if not math.isfinite(value):
            raise TraceError("Non-finite JSON number")


def validate(data):
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise TraceError("Unsupported trace schema")
    trace_id(data.get("trace_id"))
    if data.get("status") not in ("complete", "failed") or not isinstance(data.get("metadata"), dict):
        raise TraceError("Invalid trace status or metadata")
    for section in SECTIONS:
        rows = data.get(section)
        if not isinstance(rows, list) or len(rows) > MAX_ROWS or any(not isinstance(row, dict) for row in rows):
            raise TraceError(f"Invalid section: {section}")
    required = {"modules": ("name", "type"), "events": ("module", "id", "occurrence", "inputs", "outputs", "status"),
                "parameters": ("name", "requires_grad", "grad_status"), "gradients": ("objective", "target", "status"),
                "comparisons": ("target", "left", "right", "cosine"), "warnings": ("code",)}
    for section, keys in required.items():
        for row in data[section]:
            if any(k not in row for k in keys):
                raise TraceError(f"Missing row field in {section}")
    for section, keys in {"modules": ("name", "type"), "events": ("module", "status"), "parameters": ("name", "grad_status"), "gradients": ("objective", "target", "status"), "comparisons": ("target", "left", "right"), "warnings": ("code",)}.items():
        if any(not isinstance(row[k], str) for row in data[section] for k in keys):
            raise TraceError(f"Invalid field type in {section}")
    for row in data["events"]:
        if type(row["id"]) is not int or type(row["occurrence"]) is not int or not isinstance(row["inputs"], list) or not isinstance(row["outputs"], list):
            raise TraceError("Invalid event shape")
    for row in data["parameters"]:
        if type(row["requires_grad"]) is not bool:
            raise TraceError("Invalid requires_grad")
    _check(data)
    return data


def save_trace(root, data):
    validate(data)
    payload = encode(data).encode("utf-8")
    if len(payload) > MAX_BYTES:
        raise TraceError("Trace too large; reduce event or tensor limits")
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    target = root / (data["trace_id"] + ".json")
    # Hard link publishes an already complete file without overwriting a previous trace.
    fd, temp = tempfile.mkstemp(prefix=".tensorpath-", dir=root)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.link(temp, target)
    finally:
        os.unlink(temp)
    return target


class TraceStore:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise TraceError("Trace root is not a directory")

    def ids(self):
        return sorted(p.stem for p in self.root.glob("*.json") if ID.fullmatch(p.stem) and not p.is_symlink() and p.is_file())

    def load(self, identifier):
        path = self.root / (trace_id(identifier) + ".json")
        try:
            if path.is_symlink() or not path.is_file():
                raise TraceError("Trace not found or not a regular file")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
            with os.fdopen(os.open(path, flags), "rb") as f:
                st = os.fstat(f.fileno())
                if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_BYTES:
                    raise TraceError("Invalid trace file or size")
                raw = f.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise TraceError("Trace too large")
            data = validate(json.loads(raw))
            if data["trace_id"] != identifier:
                raise TraceError("Trace ID does not match filename")
            return data
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            if isinstance(exc, TraceError):
                raise
            raise TraceError("Could not read or parse trace") from exc
