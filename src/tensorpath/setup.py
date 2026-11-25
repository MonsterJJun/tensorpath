"""Explicit, backup-first Hermes setup. No shell execution or dependency installation."""
from importlib import resources, util
from pathlib import Path
from collections.abc import MutableMapping
import io
import os
import shutil
import sys
import tempfile
from .store import TraceError


def skill_sources():
    return sorted((p for p in resources.files('tensorpath_skills').iterdir() if p.is_dir() and p.joinpath('SKILL.md').is_file()), key=lambda p: p.name)


def setup_hermes(root, *, project=None, reports=None, agent_home=None, apply=False, replace=False):
    from ruamel.yaml import YAML
    root = Path(root).expanduser().resolve()
    home = Path(agent_home or Path.home()/'.hermes').expanduser().resolve(strict=True)
    if not home.is_dir():
        raise TraceError('Hermes home must exist')
    config = home/'config.yaml'
    skills = home/'skills'
    if config.is_symlink() or not config.is_file() or skills.is_symlink() or (skills.exists() and not skills.is_dir()):
        raise TraceError('Expected regular Hermes config.yaml and skills directory')
    reports = Path(reports or root/'reports').expanduser().resolve()
    project = Path(project).expanduser().resolve(strict=True) if project else None
    if project is not None and not project.is_dir():
        raise TraceError('Project must be a directory')
    for p in (root, reports):
        if p.exists() and not p.is_dir():
            raise TraceError('Trace/report roots must be directories')
    raw = config.read_bytes()
    if len(raw) > 4*1024*1024:
        raise TraceError('Config exceeds 4 MiB')
    yaml = YAML(); yaml.allow_duplicate_keys = False; yaml.preserve_quotes = True
    try:
        data = yaml.load(raw.decode('utf-8'))
    except Exception as exc:
        raise TraceError('Cannot safely parse Hermes YAML; no changes made') from exc
    if not isinstance(data, MutableMapping):
        raise TraceError('Hermes YAML root must be a mapping')
    servers = data.setdefault('mcp_servers', {})
    if not isinstance(servers, MutableMapping):
        raise TraceError('mcp_servers must be a mapping')
    args = ['-m','tensorpath.server','--root',str(root),'--reports',str(reports)]
    if project is not None:
        args += ['--project', str(project)]
    desired = {'command': os.path.abspath(sys.executable), 'args': args}
    config_changed = servers.get('tensorpath') != desired
    if 'tensorpath' in servers and config_changed and not replace:
        raise TraceError('Existing TensorPath MCP differs; review it and rerun with --replace')
    sources = skill_sources()
    if not sources:
        raise TraceError('No bundled skills found')
    changed = []
    for source in sources:
        dest = skills/source.name
        file = dest/'SKILL.md'
        if dest.is_symlink() or file.is_symlink() or (dest.exists() and not dest.is_dir()):
            raise TraceError(f'Unsafe existing skill: {source.name}')
        if file.is_file() and file.read_bytes() == source.joinpath('SKILL.md').read_bytes():
            continue
        if dest.exists() and not replace:
            raise TraceError(f'Existing skill differs: {source.name}; review and use --replace')
        changed.append(source)
    result = {'mode':'apply' if apply else 'preview', 'config':str(config), 'config_changed':config_changed, 'skills_to_install':[p.name for p in changed], 'root':str(root),'reports':str(reports),'project':str(project) if project else None,'mcp_sdk_available':util.find_spec('mcp') is not None,'backup':None}
    if not apply:
        return result
    if not result['mcp_sdk_available']:
        raise TraceError('Install TensorPath with [mcp,setup] before applying setup')
    # Serialize before modifying anything. Round-trip YAML preserves unrelated keys/comments.
    servers['tensorpath'] = desired
    stream = io.StringIO(); yaml.dump(data, stream)
    if config.read_bytes() != raw:
        raise TraceError('Config changed during setup; retry with Hermes stopped')
    root.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True)
    skills.mkdir(exist_ok=True)
    if not changed and not config_changed:
        return result
    backup = Path(tempfile.mkdtemp(prefix='tensorpath-backup-', dir=home))
    result['backup'] = str(backup)
    shutil.copy2(config, backup/'config.yaml')
    installed, moved = [], []
    stage = Path(tempfile.mkdtemp(prefix='.tensorpath-stage-', dir=home))
    try:
        for source in changed:
            dest = skills/source.name
            fresh = stage/source.name; fresh.mkdir()
            (fresh/'SKILL.md').write_bytes(source.joinpath('SKILL.md').read_bytes())
            if dest.exists():
                dest.rename(backup/source.name); moved.append(source.name)
            fresh.rename(dest); installed.append(source.name)
        if config_changed:
            if config.read_bytes() != raw:
                raise TraceError('Config changed during setup; retry with Hermes stopped')
            new = stage/'config.yaml'
            new.write_text(stream.getvalue(), encoding='utf-8')
            shutil.copymode(config, new)
            os.replace(new, config)
    except BaseException:
        # Best-effort rollback; backup retained even if rollback fails.
        for name in installed:
            shutil.rmtree(skills/name)
        for name in moved:
            (backup/name).rename(skills/name)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return result
