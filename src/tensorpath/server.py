"""MCP v1 SDK adapter. Scoped stdio transport; no torch or arbitrary execution."""
import argparse
from .store import TraceStore, TraceError, encode
from .analyzer import Analyzer


def create_server(root, project=None, reports=None):
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations
    from .agent import AgentTools
    a = Analyzer(TraceStore(root))
    agent = AgentTools(root, project, reports)
    server = FastMCP("TensorPath")
    annotation = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
    # Explicit tool signatures generate small schemas; return JSON text for broad v1 client support.
    def result(fn, *args):
        try:
            out = encode(fn(*args))
            if len(out.encode()) > 32_000:
                raise TraceError("Response exceeds budget; narrow the query")
            return out
        except (TraceError, OSError) as exc:
            return encode({"error": str(exc)})

    @server.tool(annotations=annotation)
    def list_traces(offset: int = 0, limit: int = 20) -> str:
        """List recorded trace IDs; use next_offset to continue."""
        return result(a.list_traces, offset, limit)

    @server.tool(annotations=annotation)
    def inspect_model(trace: str, section: str = "parameters", offset: int = 0, limit: int = 20) -> str:
        """Read modules or parameter/optimizer snapshot metadata."""
        return result(a.inspect_model, trace, section, offset, limit)

    @server.tool(annotations=annotation)
    def trace_forward(trace: str, module: str = "", offset: int = 0, limit: int = 20) -> str:
        """Read observed module calls and tensor metadata; optional exact module name."""
        return result(a.trace_forward, trace, module, offset, limit)

    @server.tool(annotations=annotation)
    def analyze_gradients(trace: str, section: str = "objectives", target: str = "", offset: int = 0, limit: int = 20) -> str:
        """Read objectives, cosines or parameters; optional exact target name."""
        return result(a.analyze_gradients, trace, section, target, offset, limit)

    @server.tool(annotations=annotation)
    def diagnose(trace: str, offset: int = 0, limit: int = 20) -> str:
        """List evidence-based diagnostic observations, not proven root causes."""
        return result(a.diagnose, trace, offset, limit)

    @server.tool(annotations=annotation)
    def compare_traces(left: str, right: str, section: str = "events", offset: int = 0, limit: int = 20) -> str:
        """Exact comparison of events, parameters, gradients or comparisons."""
        return result(a.compare_traces, left, right, section, offset, limit)
    @server.tool(annotations=annotation)
    def discover_project(limit: int = 20) -> str:
        """Find candidate Python model classes and entrypoints within the configured project. Static, never executes code."""
        return result(agent.discover, limit)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
    def visualize_execution(trace: str, include: list[str] | None = None) -> str:
        """Create a new offline HTML report in the configured report directory; never overwrite or launch a browser."""
        return result(agent.visualize, trace, include)
    return server


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="Explicit directory of trace files")
    p.add_argument("--project", help="Approved project directory for static discovery")
    p.add_argument("--reports", help="Approved directory for new HTML reports")
    args = p.parse_args(argv)
    try:
        server = create_server(args.root, args.project, args.reports)
    except ModuleNotFoundError as exc:
        if exc.name == "mcp":
            p.error("Install the MCP extra: pip install '.[mcp]'")
        raise
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
