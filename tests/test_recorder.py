import pytest
import torch
from torch import nn
from tensorpath import Recorder
from tensorpath.store import TraceStore
from tensorpath.analyzer import Analyzer


def test_forward_nested_and_shared(tmp_path):
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.a = nn.Linear(3, 3); self.b = self.a
        def forward(self, *, x):
            return {"hidden": [self.b(self.a(x)), None]}
    m = Model(); x = torch.randn(2, 3)
    before = m(x=x)["hidden"][0]
    with Recorder(m, tmp_path, "shared") as r:
        out = m(x=x); out["hidden"][0].sum().backward(); r.snapshot_parameters()
    d = TraceStore(tmp_path).load("shared")
    assert torch.equal(before, out["hidden"][0])
    assert [e["module"] for e in d["events"]] == ["<root>", "a", "a"]
    assert [e["occurrence"] for e in d["events"]] == [1, 1, 2]
    assert d["events"][2]["parent_id"] == 1
    assert d["events"][0]["outputs"][0]["shape"] == [2, 3]
    assert d["modules"][1]["aliases"] == ["a", "b"]
    assert len(d["parameters"]) == 2
    assert not m._forward_hooks and not m.a._forward_pre_hooks


def test_objectives_zero_unused_frozen_and_preserve_grad(tmp_path):
    m = nn.Linear(2, 1, bias=False)
    t = m.weight; u = nn.Parameter(torch.ones(1)); frozen = torch.ones(1)
    t.grad = torch.full_like(t, 9)
    with Recorder(m, tmp_path, "grad") as r:
        loss = t.sum()
        r.probe_gradients({"positive": loss, "negative": -loss, "zero": loss*0}, {"w": t, "unused": u, "frozen": frozen})
        assert torch.equal(t.grad, torch.full_like(t, 9))
        loss.backward(); r.snapshot_parameters()
    d = TraceStore(tmp_path).load("grad")
    rows = {(x["objective"], x["target"]): x for x in d["gradients"]}
    assert rows["positive", "unused"]["status"] == "unused"
    assert rows["positive", "frozen"]["status"] == "frozen"
    assert rows["zero", "w"]["status"] == "zero"
    assert rows["positive", "w"]["norm"] == pytest.approx(2**.5)
    c = {(x["left"], x["right"], x["target"]): x["cosine"] for x in d["comparisons"]}
    assert c["positive", "negative", "w"] == pytest.approx(-1)
    assert c["positive", "zero", "w"] is None


def test_detach_optimizer(tmp_path):
    m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    opt = torch.optim.SGD(m[1].parameters(), lr=.1)
    with Recorder(m, tmp_path, "detached") as r:
        y = m[1](m[0](torch.ones(1,2)).detach()).sum()
        r.probe_gradients({"loss": y}, {"first": m[0].weight})
        y.backward(); r.snapshot_parameters(opt)
    findings = Analyzer(TraceStore(tmp_path)).diagnose("detached")["items"]
    assert {"parameter_grad_missing", "trainable_not_in_optimizer", "objective_grad_unused"} <= {x["code"] for x in findings}


def test_failure_and_cleanup(tmp_path):
    m = nn.Linear(2, 2)
    with pytest.raises(RuntimeError):
        with Recorder(m, tmp_path, "failed"):
            m(torch.ones(1, 3))
    d = TraceStore(tmp_path).load("failed")
    assert d["status"] == "failed" and d["events"][0]["status"] == "unfinished"
    assert not m._forward_hooks and not m._forward_pre_hooks


def test_event_limit(tmp_path):
    m = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    with Recorder(m, tmp_path, "limited", max_events=1):
        m(torch.ones(1, 2))
    d = TraceStore(tmp_path).load("limited")
    assert len(d["events"]) == 1 and d["events"][0]["status"] == "returned"
    assert "event_limit_reached" in [x["code"] for x in d["warnings"]]


def test_uninitialized(tmp_path):
    m = nn.LazyLinear(2)
    with Recorder(m, tmp_path, "lazy") as r:
        r.snapshot_parameters()
    assert TraceStore(tmp_path).load("lazy")["parameters"][0]["grad_status"] == "uninitialized"


def test_nonfinite_and_size_limit(tmp_path):
    m = nn.Linear(2, 1)
    with Recorder(m, tmp_path, "finite", max_grad_elements=1) as r:
        loss = m.bias.sum() * float("nan")
        r.probe_gradients({"loss": loss}, {"bias": m.bias, "weight": m.weight})
    rows = TraceStore(tmp_path).load("finite")["gradients"]
    assert [x["status"] for x in rows] == ["nonfinite", "size_limit"]


def test_inference_no_grad(tmp_path):
    m = nn.Linear(2, 2)
    with Recorder(m, tmp_path, "infer"):
        with torch.no_grad():
            m(torch.ones(1, 2))
    assert TraceStore(tmp_path).load("infer")["events"][0]["outputs"][0]["requires_grad"] is False


def test_external_soft_embedding(tmp_path):
    m = nn.Linear(2, 1).requires_grad_(False)
    e = nn.Parameter(torch.ones(1, 2)); opt = torch.optim.AdamW([e])
    with Recorder(m, tmp_path, "extra") as r:
        m(e).sum().backward()
        r.snapshot_parameters(opt, extra_targets={"BE": e})
    rows = TraceStore(tmp_path).load("extra")["parameters"]
    assert rows[-1]["name"] == "BE" and rows[-1]["optimizer_groups"] == [0]
    assert rows[-1]["grad_status"] == "observed"
