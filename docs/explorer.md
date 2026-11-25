# TensorPath Explorer

## Open existing records

`python -m pip install -e .` installs the core without torch or MCP.

```sh
tensorpath --root examples/sample_traces visualize healthy --include detached --output explorer.html --open
```

The HTML contains embedded trace metadata and browser UI; it is independent of the source directory after export. `--open` asks the OS to open the file. If unavailable, open the HTML manually. An existing output cannot be overwritten; choose a new filename. Use the file picker to open further schema-v1 JSON records. IDs must be unique. At most 12 traces, 16 MiB per uploaded file and 48 MiB combined encoded trace data are accepted; view pages contain up to 100 records. No additional server, JavaScript package, external font or CDN is used. Data is not sent to a service. The HTML itself contains trace metadata: treat it as sensitive if names or paths are sensitive.

## Views

- Execution: entry order, replay/step/seek, search, selected input/output tensors, parent-child module hierarchy. The glyph is schematic; it does not represent values or exact physical shape. A parent may return only after its children; the inspector shows the recorded final output, not live state at that entry.
- Gradients: per-target objective statuses, norm bars, pairwise cosine. No exact backward graph; undefined is not zero.
- Parameters: snapshot labels, trainability, gradient status, optimizer membership. No parameter snapshot means unavailable information, not a model with no weights.
- Compare: select another trace and a section, click a changed key to inspect left/right fields. No statistical or causal conclusion is implied. Browser numbers are IEEE-754 double precision; use the Python CLI for authoritative large-integer or exact metadata comparisons.

## Capture without training

```sh
python -m pip install -e '.[record]'
python examples/forward_only.py --root traces_forward
tensorpath --root traces_forward visualize cnn_forward --include lstm_forward --include branched_forward --output forward.html
```

The examples use freshly initialized CNN, LSTM and branched modules and synthetic input with `no_grad()`. No gradient probes or optimizer steps are run. `nn.LSTM` may appear as one module. The branch addition is functional and is not recorded as an operator. Module parent edges must not be mistaken for tensor dependencies.

## Developer checks

```sh
python -m pip install -e '.[test]'
python -m pytest -q
python -m pip install playwright
python -m playwright install chromium
python tests/browser_smoke.py
```

The browser test is optional; it is separate from the core dependency set. Use `--browser-path` for a system Chromium. The test explicitly reports a content-injection fallback when a managed browser blocks local file navigation. See the verification report for actual executed checks and limitations.
