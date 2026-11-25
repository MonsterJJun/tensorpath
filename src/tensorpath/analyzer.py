"""Deterministic, read-only analysis; never imports torch or runs model code."""
from collections import Counter
from .store import TraceError, encode

RESPONSE_BYTES = 24_000


def page(rows, offset=0, limit=20):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
        raise TraceError("offset >= 0 and 1 <= limit <= 100 required")
    result = []
    used = 0
    for row in rows[offset:offset + limit]:
        size = len(encode(row).encode())
        if used + size > RESPONSE_BYTES:
            if not result:
                result.append({"omitted": "row exceeds response budget", "index": offset})
            break
        result.append(row)
        used += size
    end = offset + len(result)
    return {"total": len(rows), "offset": offset, "items": result, "next_offset": end if end < len(rows) else None}


class Analyzer:
    def __init__(self, store):
        self.store = store

    def list_traces(self, offset=0, limit=20):
        return page([{"trace_id": name} for name in self.store.ids()], offset, limit)

    def inspect_model(self, trace, section="parameters", offset=0, limit=20):
        d = self.store.load(trace)
        if section not in ("modules", "parameters"):
            raise TraceError("section must be modules or parameters")
        rows = d["parameters"]
        return {"trace_id": trace, "status": d["status"], "snapshot": d["metadata"].get("parameter_snapshot"),
                "module_count": len(d["modules"]), "parameter_tensor_count": len(rows),
                "trainable_tensor_count": sum(r["requires_grad"] for r in rows), **page(d[section], offset, limit)}

    def trace_forward(self, trace, module="", offset=0, limit=20):
        d = self.store.load(trace)
        rows = [r for r in d["events"] if not module or r["module"] == module]
        return {"trace_id": trace, "status": d["status"], "warning_counts": dict(Counter(r["code"] for r in d["warnings"])),
                "timing": "inclusive host wall time, not GPU kernel latency", **page(rows, offset, limit)}

    def analyze_gradients(self, trace, section="objectives", target="", offset=0, limit=20):
        d = self.store.load(trace)
        names = {"objectives": "gradients", "cosines": "comparisons", "parameters": "parameters"}
        if section not in names:
            raise TraceError("section must be objectives, cosines or parameters")
        rows = d[names[section]]
        rows = [r for r in rows if not target or r.get("target", r.get("name")) == target]
        return {"trace_id": trace, "snapshot": d["metadata"].get("parameter_snapshot"),
                "status_counts": dict(Counter(r.get("grad_status", r.get("status", "comparison")) for r in rows)),
                "note": "Missing or zero gradient alone does not establish a bug; inspect snapshot timing and graph usage.", **page(rows, offset, limit)}

    def diagnose(self, trace, offset=0, limit=20):
        d = self.store.load(trace)
        findings = [{"code": r["code"], "evidence": "warnings"} for r in d["warnings"]]
        if d["status"] == "failed":
            findings.append({"code": "run_failed", "exception_type": d["metadata"].get("exception_type")})
        if d["metadata"].get("parameter_snapshot") == "not_captured":
            findings.append({"code": "parameter_snapshot_missing"})
        for p in d["parameters"]:
            if p["requires_grad"] and p.get("optimizer_groups") == []:
                findings.append({"code": "trainable_not_in_optimizer", "parameter": p["name"]})
            if p["grad_status"] in ("nonfinite", "missing", "zero") and p["requires_grad"]:
                findings.append({"code": "parameter_grad_" + p["grad_status"], "parameter": p["name"]})
            if not p["requires_grad"] and p.get("optimizer_groups"):
                findings.append({"code": "frozen_in_optimizer", "parameter": p["name"]})
        for g in d["gradients"]:
            if g["status"] not in ("observed",):
                findings.append({"code": "objective_grad_" + g["status"], "target": g["target"], "objective": g["objective"]})
        return {"trace_id": trace, "note": "Observations, not root-cause proof; empty findings do not certify correctness.", **page(findings, offset, limit)}

    def compare_traces(self, left, right, section="events", offset=0, limit=20):
        a, b = self.store.load(left), self.store.load(right)
        specs = {"events": (("module", "occurrence"), ("inputs", "outputs", "status", "parent_id")),
                 "parameters": (("name",), ("shape", "dtype", "device", "requires_grad", "grad_status", "grad_norm", "optimizer_groups")),
                 "gradients": (("objective", "target"), ("status", "norm")),
                 "comparisons": (("target", "left", "right"), ("cosine",))}
        if section not in specs:
            raise TraceError("Unknown comparison section")
        keys, fields = specs[section]
        def index(d):
            out = {}
            for row in d[section]:
                key = tuple(row[k] for k in keys)
                if key in out:
                    raise TraceError("Duplicate comparison key")
                out[key] = row
            return out
        x, y = index(a), index(b)
        changes = []
        for key in sorted(x.keys() | y.keys()):
            if key not in x or key not in y:
                changes.append({"key": list(key), "change": "added" if key not in x else "removed"})
            else:
                diff = {f: {"left": x[key].get(f), "right": y[key].get(f)} for f in fields if x[key].get(f) != y[key].get(f)}
                if diff:
                    changes.append({"key": list(key), "change": "modified", "fields": diff})
        return {"left": left, "right": right, "left_status": a["status"], "right_status": b["status"],
                "note": "Exact metadata comparison; control inputs, seeds, graph and snapshot stage separately.", **page(changes, offset, limit)}
