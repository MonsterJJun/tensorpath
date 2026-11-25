import json
from pathlib import Path
import pytest
from tensorpath.agent import AgentTools
from tensorpath.discovery import discover_project
from tensorpath.setup import setup_hermes, skill_sources
from tensorpath.store import TraceError, save_trace
from tensorpath.doctor import doctor
from test_tensorpath import sample


def test_discovery_static(tmp_path):
    (tmp_path/'model.py').write_text('raise RuntimeError("never execute")\nclass Net(nn.Module):\n def forward(self, x): return x\n')
    (tmp_path/'train.py').write_text('if __name__ == "__main__":\n pass\n')
    (tmp_path/'broken.py').write_text('def ???')
    (tmp_path/'.hidden').mkdir(); (tmp_path/'.hidden'/'model.py').write_text('class Hidden(Module): pass')
    result = discover_project(tmp_path)
    assert [r['path'] for r in result['items']] == ['model.py','train.py']
    assert result['items'][0]['model_candidates'][0]['name'] == 'Net'
    assert result['items'][1]['main_guard'] and result['files_or_dirs_skipped'] == 1


def test_discovery_scope_and_limits(tmp_path):
    with pytest.raises(TraceError): discover_project(None)
    with pytest.raises(TraceError): discover_project(tmp_path, 0)
    for n in range(3): (tmp_path/f'model{n}.py').write_text('class M(Module): pass')
    result = discover_project(tmp_path,1)
    assert len(result['items']) == 1 and result['truncated']


def test_visualize_agent(tmp_path):
    save_trace(tmp_path, sample())
    reports = tmp_path/'reports'; reports.mkdir()
    a = AgentTools(tmp_path, reports=reports)
    p = Path(a.visualize('demo')['path']); q = Path(a.visualize('demo')['path'])
    assert p.parent == reports and p != q and p.is_file() and q.is_file()
    with pytest.raises(TraceError): a.visualize('../escape')
    with pytest.raises(TraceError): a.visualize('demo',['demo'])
    with pytest.raises(TraceError): AgentTools(tmp_path).visualize('demo')


def home_fixture(tmp_path):
    home = tmp_path/'hermes'; home.mkdir()
    (home/'config.yaml').write_text('# preserve comment\nmodel: custom\nmcp_servers:\n  research_rag:\n    url: http://localhost:8765/mcp\n')
    return home


def test_setup_preview_no_mutation(tmp_path):
    home = home_fixture(tmp_path); root=tmp_path/'traces'
    before=(home/'config.yaml').read_bytes()
    info=setup_hermes(root,agent_home=home,project=tmp_path)
    assert len(info['skills_to_install'])==10 and not root.exists() and not (home/'skills').exists()
    assert (home/'config.yaml').read_bytes()==before


def test_setup_apply_backup_idempotent(tmp_path, monkeypatch):
    import tensorpath.setup as s
    monkeypatch.setattr(s.util,'find_spec',lambda n: True)
    home=home_fixture(tmp_path); root=tmp_path/'traces'
    before=(home/'config.yaml').read_bytes()
    info=setup_hermes(root,agent_home=home,project=tmp_path,apply=True)
    assert (Path(info['backup'])/'config.yaml').read_bytes()==before
    updated=(home/'config.yaml').read_text()
    assert '# preserve comment' in updated and 'research_rag' in updated and 'tensorpath.server' in updated
    assert len(list((home/'skills').glob('*/SKILL.md')))==10
    again=setup_hermes(root,agent_home=home,project=tmp_path,apply=True)
    assert not again['config_changed'] and not again['skills_to_install'] and again['backup'] is None


def test_setup_conflict_replace(tmp_path, monkeypatch):
    import tensorpath.setup as s
    monkeypatch.setattr(s.util,'find_spec',lambda n: True)
    home=home_fixture(tmp_path); root=tmp_path/'traces'
    dest=home/'skills'/'tensorpath'; dest.mkdir(parents=True); (dest/'SKILL.md').write_text('custom')
    with pytest.raises(TraceError): setup_hermes(root,agent_home=home,apply=True)
    assert (dest/'SKILL.md').read_text()=='custom'
    info=setup_hermes(root,agent_home=home,apply=True,replace=True)
    assert (Path(info['backup'])/'tensorpath'/'SKILL.md').read_text()=='custom'
    assert (dest/'SKILL.md').read_text()!='custom'


def test_setup_duplicate_yaml_rejected(tmp_path):
    home=home_fixture(tmp_path); (home/'config.yaml').write_text('mcp_servers: {}\nmcp_servers: {}\n')
    with pytest.raises(TraceError): setup_hermes(tmp_path/'traces',agent_home=home)


def test_doctor_and_skills(tmp_path):
    assert len(skill_sources())==10
    info=doctor(tmp_path)
    assert info['protocol']=='not_requested' and info['version']=='0.2.1'


def test_forward_capture_no_parameter_update(tmp_path):
    torch=pytest.importorskip('torch')
    from tensorpath import capture_forward
    model=torch.nn.Linear(2,3); before=model.weight.detach().clone()
    p=capture_forward(model,tmp_path,args=(torch.ones(1,2),))
    d=json.loads(p.read_text()); assert d['events'][0]['outputs'][0]['shape']==[1,3]
    assert torch.equal(before,model.weight) and model.weight.grad is None and model.training


def test_setup_rolls_back_skills_on_config_failure(tmp_path, monkeypatch):
    import tensorpath.setup as s
    monkeypatch.setattr(s.util,'find_spec',lambda n: True)
    home=home_fixture(tmp_path); root=tmp_path/'traces'
    dest=home/'skills'/'tensorpath'; dest.mkdir(parents=True); (dest/'SKILL.md').write_text('custom')
    before=(home/'config.yaml').read_bytes()
    def fail(*_): raise OSError('simulated replace failure')
    monkeypatch.setattr(s.os,'replace',fail)
    with pytest.raises(OSError): setup_hermes(root,agent_home=home,apply=True,replace=True)
    assert (dest/'SKILL.md').read_text()=='custom'
    assert (home/'config.yaml').read_bytes()==before
    assert len(list((home/'skills').glob('*/SKILL.md')))==1


def test_setup_missing_sdk_no_changes(tmp_path, monkeypatch):
    import tensorpath.setup as s
    monkeypatch.setattr(s.util,'find_spec',lambda n: None)
    home=home_fixture(tmp_path); root=tmp_path/'traces'
    before=(home/'config.yaml').read_bytes()
    with pytest.raises(TraceError, match='Install TensorPath'):
        setup_hermes(root,agent_home=home,apply=True)
    assert not root.exists() and not (home/'skills').exists()
    assert (home/'config.yaml').read_bytes()==before
