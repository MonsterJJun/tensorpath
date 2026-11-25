"""Real protocol test when optional MCP SDK is installed."""
import asyncio
import json
import os
import sys
import pytest

mcp = pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tensorpath.store import save_trace
from test_tensorpath import sample


def test_stdio_roundtrip(tmp_path):
    save_trace(tmp_path, sample())
    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-m", "tensorpath.server", "--root", str(tmp_path), "--reports", str(tmp_path), "--project", str(tmp_path)], env=dict(os.environ))
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert len(tools.tools) == 8
                result = await session.call_tool("list_traces", {})
                assert not result.isError
                assert json.loads(result.content[0].text)["items"] == [{"trace_id": "demo"}]
                vis = await session.call_tool("visualize_execution", {"trace":"demo"})
                assert not vis.isError
                assert os.path.isfile(json.loads(vis.content[0].text)["path"])
                discovery = await session.call_tool("discover_project", {})
                assert not discovery.isError
                assert "items" in json.loads(discovery.content[0].text)
                invalid = await session.call_tool("inspect_model", {"trace": "../secret"})
                assert "error" in json.loads(invalid.content[0].text)
    asyncio.run(asyncio.wait_for(run(), timeout=30))
