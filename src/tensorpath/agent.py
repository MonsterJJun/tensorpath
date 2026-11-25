"""Agent-facing capabilities: scoped static inventory and new HTML artifacts."""
from pathlib import Path
from uuid import uuid4
from .store import TraceError, trace_id
from .visualizer import export_html
from .discovery import discover_project


class AgentTools:
    def __init__(self, root, project=None, reports=None):
        self.root = Path(root).resolve(strict=True)
        self.project = Path(project).resolve(strict=True) if project is not None else None
        self.reports = Path(reports).resolve(strict=True) if reports is not None else None
        if any(not p.is_dir() for p in (self.project, self.reports) if p is not None):
            raise TraceError('Configured project and report roots must be directories')

    def discover(self, limit=20):
        return discover_project(self.project, limit)

    def visualize(self, trace, include=None):
        if self.reports is None:
            raise TraceError('No report directory configured; use setup --reports or server --reports')
        trace_id(trace)
        if include is not None and not isinstance(include, list):
            raise TraceError('include must be a list of trace IDs')
        for item in include or []:
            trace_id(item)
        name = f'{trace}-{uuid4().hex[:12]}.html'
        path = export_html(self.root, trace, include=include or [], output=self.reports / name)
        return {'path': str(path), 'trace_ids': [trace, *(include or [])], 'format': 'offline_html', 'coverage': 'module_calls_and_recorded_metadata_not_operator_dataflow', 'browser_opened': False}
