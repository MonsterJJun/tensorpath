"""Check this installation and optionally perform a real MCP stdio handshake."""
import asyncio
import importlib.util
import os
import json
import sys
from pathlib import Path
from . import __version__
from .store import TraceStore

EXPECTED = {'list_traces','inspect_model','trace_forward','analyze_gradients','diagnose','compare_traces','discover_project','visualize_execution'}


def doctor(root, project=None, reports=None, protocol=False):
    store = TraceStore(root)
    result = {'version':__version__, 'python':sys.executable, 'trace_count':len(store.ids()), 'mcp_sdk_available':importlib.util.find_spec('mcp') is not None, 'protocol':'not_requested'}
    if protocol:
        if not result['mcp_sdk_available']:
            result['protocol'] = 'missing_sdk'; return result
        async def check():
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            args = ['-m','tensorpath.server','--root',str(store.root)]
            for flag, value in (('--project', project),('--reports', reports)):
                if value is not None: args.extend([flag,str(Path(value).expanduser().resolve())])
            async with stdio_client(StdioServerParameters(command=sys.executable,args=args,env=dict(os.environ))) as (reader, writer):
                async with ClientSession(reader,writer) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    names = {t.name for t in tools.tools}
                    response = await session.call_tool('list_traces', {'limit':1})
                    payload = json.loads(response.content[0].text)
                    return names, response.isError or not isinstance(payload.get("items"), list)
        try:
            names, error = asyncio.run(asyncio.wait_for(check(), timeout=20))
            result['protocol'] = 'passed' if EXPECTED == names and not error else 'failed'
            result['tools'] = sorted(names)
        except Exception as exc:
            result['protocol'] = 'failed'; result['exception_type'] = type(exc).__name__
    return result
