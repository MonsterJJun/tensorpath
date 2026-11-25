---
name: tensorpath-record-execution
description: Prepare and collect minimal PyTorch runtime evidence using an approved existing model and safe execution plan.
---

1. Reuse an existing trace when possible. Otherwise inspect the target, its constructor, input contract, device and environment. Decide whether a forward-only run, loss probe, or actual optimizer update is necessary. No training or weights download by default.
2. Present the precise proposed script, command, working directory, dependencies, data/weights, output directory and potential side effects. Obtain explicit approval before importing or executing project code, modifying files, downloading, or starting training. No automatic escalation to arbitrary execution. `no_grad` does not prevent forward side effects (e.g. mutable buffers, custom I/O).
3. In the approved project Python environment (not necessarily the MCP environment), use TensorPath's `capture_forward` helper with an existing model and input. Example:

```python
from tensorpath import capture_forward
path = capture_forward(model, approved_trace_root, args=(x,))
```

Use kwargs for named model inputs. Choose eval mode explicitly if appropriate; the helper preserves training flags and does not call backward or optimizer.step. Prefer a small synthetic input only when semantically valid. For gradients use `Recorder.probe_gradients` with explicit scalar losses and named targets before backward; snapshot after backward, before zero_grad. Do not claim updates from gradient snapshots.

4. Run via the host's terminal tool only with approval, using a temporary or new script rather than changing the training source by default. If the model is unavailable or unsafe to execute, stop with a static explanation or request a trace. MCP has no execution endpoint.
5. Confirm the new trace exists via `list_traces`, check status/coverage/limits, and report failures explicitly. No successful run, no runtime claims.
