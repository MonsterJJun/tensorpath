import json
import math
from pathlib import Path
import pytest

from tensorpath.store import TraceStore, TraceError, save_trace
from tensorpath.analyzer import Analyzer, page
from tensorpath.cli import main


def sample(name="demo"):
    return {"schema_version": 1, "trace_id": name, "status": "complete", "metadata": {"parameter_snapshot": "not_captured"},
            **{k: [] for k in ("modules", "events", "parameters", "gradients", "comparisons", "warnings")}}


def test_roundtrip_no_overwrite(tmp_path):
    d = sample()
    save_trace(tmp_path, d)
    assert TraceStore(tmp_path).load("demo") == d
    with pytest.raises(FileExistsError):
        save_trace(tmp_path, d)
    assert TraceStore(tmp_path).ids() == ["demo"]


@pytest.mark.parametrize("bad", ["../secret", "/etc/passwd", "a/b", "a.json", "..", ""])
def test_path_rejection(tmp_path, bad):
    with pytest.raises(TraceError):
        TraceStore(tmp_path).load(bad)


def test_symlink(tmp_path):
    p = tmp_path / "other.json"
    p.write_text(json.dumps(sample("link")))
    try:
        (tmp_path / "link.json").symlink_to(p)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(TraceError):
        TraceStore(tmp_path).load("link")
    assert "link" not in TraceStore(tmp_path).ids()


@pytest.mark.parametrize("bad", ["{", '{"schema_version": 5}', "[]", json.dumps(sample("different"))])
def test_invalid_file(tmp_path, bad):
    (tmp_path / "demo.json").write_text(bad)
    with pytest.raises(TraceError):
        TraceStore(tmp_path).load("demo")


def test_nonfinite(tmp_path):
    d = sample(); d["metadata"]["x"] = float("nan")
    with pytest.raises(TraceError):
        save_trace(tmp_path, d)


def test_page_budget():
    r = page([{"text": "x" * 30000}, {"small": True}])
    assert r["items"][0]["omitted"] and r["next_offset"] == 1
    assert page([1, 2, 3], 1, 1)["items"] == [2]
    with pytest.raises(TraceError):
        page([], 0, 101)


def test_cli(tmp_path, capsys):
    save_trace(tmp_path, sample())
    assert main(["--root", str(tmp_path), "list"]) == 0
    assert json.loads(capsys.readouterr().out)["items"] == [{"trace_id": "demo"}]
    assert main(["--root", str(tmp_path), "inspect", "missing"]) == 2


def test_diagnosis_and_compare(tmp_path):
    a, b = sample("a"), sample("b")
    for d, status in ((a, "observed"), (b, "missing")):
        d["parameters"] = [{"name": "w", "requires_grad": True, "grad_status": status, "optimizer_groups": []}]
        save_trace(tmp_path, d)
    api = Analyzer(TraceStore(tmp_path))
    codes = {r["code"] for r in api.diagnose("b")["items"]}
    assert {"parameter_grad_missing", "trainable_not_in_optimizer"} <= codes
    diff = api.compare_traces("a", "b", "parameters")["items"]
    assert diff[0]["fields"]["grad_status"] == {"left": "observed", "right": "missing"}
