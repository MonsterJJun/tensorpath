# Trace contract (schema 1)

One JSON file per trace ID (`[A-Za-z0-9][A-Za-z0-9_-]{0,79}`). IDs are immutable on publication; write another ID to record another run. `status=complete` means the recorder context returned without a propagated exception, not that a training step succeeded scientifically. `failed` includes exception type only. Unfinished calls and warnings may occur even in complete traces.

- `metadata`: runtime versions, model type, capture limits, snapshot label and whether optimizer membership was checked.
- `modules`: canonical name, aliases, class, training flag at registration.
- `events`: ascending entry ID, parent ID, module, occurrence, nested input/output tensor metadata, returned/unfinished status, inclusive host elapsed time for returned calls. Container path uses positional index for mappings to avoid retaining mapping keys. Parent ID is nesting, not tensor dataflow.
- `parameters`: unique model parameters plus explicitly supplied external leaf tensors, aliases, shape/dtype/device, requires_grad, grad_status/norm, optimizer group indexes (`null` if unchecked, `[]` if absent). Snapshot reflects a single instant, not actual optimizer updates.
- `gradients`: objective and target name, status, L2 norm or null. `unused` means autograd returned None; `zero` means it returned a zero vector. `loss_no_grad`, `frozen`, `autograd_error`, `unsupported_layout`, `size_limit`, `nonfinite`, `unavailable` distinguish unmeasured cases. Gradients are independent probes, not Adam updates.
- `comparisons`: target, pair of objective names, cosine (null if undefined or unavailable). Norm and cosine use CPU float64 for selected bounded targets. Sign and scaling of supplied loss are preserved.
- `warnings`: capture limitation codes.

Limits: 16 MiB per file; 20,000 records per section; maximum 5,000 events (default 1,000), 64 tensor descriptions per input/output (default 16), 1M elements per gradient (default 1M), 4M total selected objective-target elements per probe; max 8 objectives and 32 targets. Large parameter gradients are skipped rather than copied. A trace can exceed the file-size cap: reduce limits, use a smaller debug batch, or narrow the model. Publication requires a filesystem supporting hard links (APFS, NTFS, common Linux filesystems). Use a local disk, not an untested sync/network filesystem.

Comparison matches `(module, occurrence)` for events; it is not graph isomorphism or an alignment of semantically different models. Timing excluded from comparison. No tolerance is applied; agents should assess practical magnitude. Repeated calls to snapshot or probe replace their previous sections. Read tools return `total`, `offset`, `items`, `next_offset`; default 20 rows, max 100. Item payload budget ~24 KB; MCP response capped at 32 KB JSON before protocol wrapping. A single oversized row is marked omitted. CLI does not enforce outer MCP budget. Validate the current trace ID and status before drawing conclusions.
