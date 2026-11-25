"""Local CLI for agents without an MCP transport."""
import argparse
import json
import sys
from .store import TraceStore, TraceError
from .analyzer import Analyzer


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tensorpath")
    parser.add_argument("--root", required=True, help="Directory containing trace JSON files")
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("list", "inspect", "forward", "gradients", "diagnose", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--offset", type=int, default=0)
        p.add_argument("--limit", type=int, default=20)
        if name == "compare":
            p.add_argument("left"); p.add_argument("right")
            p.add_argument("--section", default="events", choices=("events", "parameters", "gradients", "comparisons"))
        elif name != "list":
            p.add_argument("trace")
            if name == "inspect":
                p.add_argument("--section", default="parameters", choices=("parameters", "modules"))
            elif name == "forward":
                p.add_argument("--module", default="")
            elif name == "gradients":
                p.add_argument("--section", default="objectives", choices=("objectives", "cosines", "parameters"))
                p.add_argument("--target", default="")
    vis = sub.add_parser("visualize", help="Export an offline interactive HTML trace explorer")
    vis.add_argument("trace", nargs="?", help="Trace ID (defaults to first available)")
    vis.add_argument("--include", action="append", default=[], help="Embed an additional trace (repeatable)")
    vis.add_argument("--output", default="tensorpath-explorer.html")
    vis.add_argument("--open", action="store_true", dest="open_browser")
    setup = sub.add_parser("setup", help="Preview/apply Hermes skill and MCP registration")
    setup.add_argument("--project")
    setup.add_argument("--reports")
    setup.add_argument("--agent-home")
    setup.add_argument("--apply", action="store_true")
    setup.add_argument("--replace", action="store_true", help="Back up and replace conflicting TensorPath entries only")
    doc = sub.add_parser("doctor", help="Check environment and optionally MCP protocol")
    doc.add_argument("--project")
    doc.add_argument("--reports")
    doc.add_argument("--protocol", action="store_true")
    args = vars(parser.parse_args(argv))
    try:
        action = args.pop("action")
        if action == "setup":
            from .setup import setup_hermes
            print(json.dumps(setup_hermes(**args), indent=2))
            return 0
        if action == "doctor":
            from .doctor import doctor
            info = doctor(**args)
            print(json.dumps(info, indent=2))
            return 2 if info['protocol'] in ('failed','missing_sdk') else 0
        if action == "visualize":
            from .visualizer import export_html
            print(export_html(args.pop("root"), args.pop("trace"), **args))
            return 0
        a = Analyzer(TraceStore(args.pop("root")))
        method = {"list": a.list_traces, "inspect": a.inspect_model, "forward": a.trace_forward,
                  "gradients": a.analyze_gradients, "diagnose": a.diagnose, "compare": a.compare_traces}[action]
        print(json.dumps(method(**args), ensure_ascii=True, indent=2, allow_nan=False))
        return 0
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("ruamel"):
            print("TensorPath: install the setup extra: pip install '.[setup]'", file=sys.stderr)
            return 2
        raise
    except (TraceError, OSError) as exc:
        print(f"TensorPath: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
