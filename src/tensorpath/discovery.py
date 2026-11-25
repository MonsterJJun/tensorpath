"""Bounded, static Python project inventory. Never imports project code."""
import ast
import os
from pathlib import Path
from .store import TraceError

SKIP = {'node_modules', 'venv', 'env', '__pycache__', 'build', 'dist', 'site-packages', 'data', 'datasets', 'checkpoints', 'traces'}


def discover_project(root, limit=20):
    if root is None:
        raise TraceError('No project root configured; use setup --project or server --project')
    if type(limit) is not int or not 1 <= limit <= 40:
        raise TraceError('limit must be 1–40')
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise TraceError('Project root is not a directory')
    candidates, scanned, skipped, visited = [], 0, 0, 0
    truncated = False
    def walk_error(_):
        nonlocal skipped
        skipped += 1
    for folder, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in SKIP and not (Path(folder)/d).is_symlink())
        visited += len(dirs) + len(names) + 1
        if visited > 4000:
            truncated = True
            break
        for name in sorted(names):
            if not name.endswith('.py') or name.startswith('.'):
                continue
            if scanned >= 200 or len(candidates) >= limit:
                truncated = True
                break
            p = Path(folder)/name
            scanned += 1
            try:
                if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):
                    skipped += 1
                    continue
                with p.open('rb') as f:
                    raw = f.read(256_001)
                if len(raw) > 256_000:
                    skipped += 1
                    continue
                tree = ast.parse(raw, filename=name)
            except (OSError, SyntaxError, ValueError, RecursionError, UnicodeError):
                skipped += 1
                continue
            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and (any(isinstance(b, ast.Name) and b.id in {'Module','PreTrainedModel'} or isinstance(b, ast.Attribute) and b.attr in {'Module','PreTrainedModel'} for b in node.bases) or any(isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == 'forward' for m in node.body)):
                    classes.append({'name': node.name[:120], 'line': node.lineno})
            entry = any(isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name) and n.test.left.id == '__name__' and any(isinstance(v, ast.Constant) and v.value == '__main__' for v in n.test.comparators) for n in tree.body)
            if classes or entry or p.stem in {'train','model','main','inference','predict','finetune'}:
                candidates.append({'path': p.relative_to(root).as_posix()[:512], 'model_candidates': classes[:12], 'main_guard': entry, 'class_list_truncated': len(classes)>12})
        if truncated:
            break
    return {'observation': 'static_ast_candidates_not_runtime_evidence', 'items': candidates, 'files_scanned': scanned, 'files_or_dirs_skipped': skipped, 'truncated': truncated, 'limits': {'files': 200, 'entries': 4000, 'file_bytes': 256000}}
