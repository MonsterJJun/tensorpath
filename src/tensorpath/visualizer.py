"""Offline, read-only HTML export; no torch, server or CDN required."""
from importlib.resources import files
from pathlib import Path
import webbrowser
from .store import TraceStore, TraceError, encode


def export_html(root, identifier=None, *, output='tensorpath-explorer.html', open_browser=False, include=()):
    store = TraceStore(root)
    if identifier is None:
        ids = store.ids()
        identifier = ids[0] if ids else None
    selected = ([identifier] if identifier is not None else []) + list(include or ())
    if len(selected) > 12 or len(set(selected)) != len(selected):
        raise TraceError('Select at most 12 distinct traces')
    traces = [store.load(name) for name in selected]
    if sum(len(encode(d).encode('utf-8')) for d in traces) > 48 * 1024 * 1024:
        raise TraceError('Combined trace size exceeds 48 MiB')
    trace = traces if len(traces) > 1 else traces[0] if traces else None
    # Escape HTML raw-text delimiters; JSON remains valid and inert.
    payload = encode(trace).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = files('tensorpath').joinpath('web', 'explorer.html').read_text(encoding='utf-8')
    if template.count('@@TRACE_JSON@@') != 1:
        raise TraceError('Invalid explorer template')
    requested = Path(output).expanduser()
    path = requested.parent.resolve() / requested.name
    # No overwrite, including source traces. Explicit destination is required for a second export.
    with path.open('x', encoding='utf-8') as stream:
        stream.write(template.replace('@@TRACE_JSON@@', payload))
    if open_browser:
        webbrowser.open(path.as_uri())
    return path
