import subprocess
import sys
from pathlib import Path


def test_analyzer_no_torch_import():
    subprocess.run([sys.executable, "-c", "import sys; import tensorpath.analyzer, tensorpath.server; assert 'torch' not in sys.modules"], check=True)


def test_skill_manifest():
    root = Path(__file__).resolve().parents[1] / "skills"
    skills = sorted(root.glob("*/SKILL.md"))
    assert len(skills) == 10
    for path in skills:
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "---" and lines[1] == f"name: {path.parent.name}"
        assert lines[2].startswith("description: ") and lines[3] == "---"
