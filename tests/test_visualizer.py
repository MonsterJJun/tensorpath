import json
import re
from pathlib import Path
import pytest
from tensorpath.visualizer import export_html
from tensorpath.store import save_trace, TraceError

SAMPLES = Path(__file__).resolve().parents[1] / 'examples/sample_traces'


def test_export_roundtrip(tmp_path):
    out = export_html(SAMPLES, 'healthy', output=tmp_path/'view.html')
    s = out.read_text()
    data = re.search(r'<script id="initial" type="application/json">(.*?)</script>', s, re.S).group(1)
    assert json.loads(data) == json.loads((SAMPLES/'healthy.json').read_text())
    assert 'default-src' in s
    with pytest.raises(FileExistsError):
        export_html(SAMPLES, 'healthy', output=out)


def test_escape_untrusted_trace(tmp_path):
    d = json.loads((SAMPLES/'healthy.json').read_text())
    d['metadata']['model_type'] = '</script><script>alert(1)</script>'
    save_trace(tmp_path, d)
    out = export_html(tmp_path, 'healthy', output=tmp_path/'view.html')
    assert d['metadata']['model_type'] not in out.read_text()
    assert '\\u003c/script' in out.read_text()


def test_empty_root_and_bad_id(tmp_path):
    out = export_html(tmp_path, output=tmp_path/'empty.html')
    assert 'type="application/json">null</script>' in out.read_text()
    with pytest.raises(TraceError):
        export_html(tmp_path, '../private', output=tmp_path/'bad.html')


def test_cli_export(tmp_path):
    from tensorpath.cli import main
    out = tmp_path/'view.html'
    assert main(['--root', str(SAMPLES), 'visualize', 'healthy', '--output', str(out)]) == 0
    assert out.is_file()
    assert main(['--root', str(SAMPLES), 'visualize', 'healthy', '--output', str(out)]) == 2


def test_multiple_embedded_traces(tmp_path):
    out = export_html(SAMPLES, 'healthy', include=['detached'], output=tmp_path/'multi.html')
    data = re.search(r'<script id="initial" type="application/json">(.*?)</script>', out.read_text(), re.S).group(1)
    assert [d['trace_id'] for d in json.loads(data)] == ['healthy', 'detached']
    with pytest.raises(TraceError):
        export_html(SAMPLES, 'healthy', include=['healthy'], output=tmp_path/'bad.html')


def test_export_does_not_follow_output_symlink(tmp_path):
    out = tmp_path/'link.html'; target = tmp_path/'other.html'
    try:
        out.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip('symlink unavailable')
    with pytest.raises(FileExistsError):
        export_html(SAMPLES, 'healthy', output=out)
    assert not target.exists()
