---
name: tensorpath-visualize-execution
description: Create an offline interactive HTML explorer from recorded model executions without loading the model again.
---

1. Identify trace IDs from `list_traces`. A request for visualization authorizes creation of a new report in the configured report directory; if not configured, ask for the output scope. Include a comparison trace only when requested or useful to the stated goal.
2. Prefer `visualize_execution(trace, include)`; it generates a new uniquely named HTML file, never opens a browser and accepts no arbitrary output path. Check the returned path and report the actual trace IDs.
3. Fallback using the agent's terminal tool: `tensorpath --root <approved-trace-dir> visualize <trace-id> --output <approved-new-html-file>`. Do not overwrite files or launch a browser unless requested.
4. Explain call hierarchy vs tensor dataflow, gradient statistics vs backward graph, and absence of unrecorded tensor values. Report errors rather than inventing an artifact. No new model run is needed if trace evidence exists.
