# Verification — v0.1.0 preview

## Executed checks

Environment: Linux x86_64, Python 3.13.5, PyTorch 2.10.0+cpu. No physical Apple/CUDA/Windows system or Hermes instance was available.

`PYTHONPATH=src python -m pytest -q -rs`: **27 passed, 1 skipped**. The skipped test is the real MCP stdio client/server round-trip because `mcp` is unavailable in this environment. An attempted dependency installation did not find an available package. Do not interpret a skipped test as a pass.

- Trace storage: round trip, immutable publication, traversal and direct symlink rejection, invalid file rejection, finite numbers, bounded result paging.
- Recorder: nested keyword inputs and structured outputs, shared-module aliases and occurrence counters, parameter deduplication, unchanged forward results, hook cleanup, failure trace, capture limit, uninitialized parameters, no-grad inference.
- Gradient probes: observed, unused, frozen, zero, nonfinite, size limit, negative cosine, undefined cosine; pre-existing `.grad` preserved in tested case; external standalone trainable target and optimizer membership.
- Analyzer: CLI valid/invalid requests, diagnosis, exact comparison, independent import without torch.
- Skills: all seven folders have matching metadata and nonempty descriptions. This does not test LLM adherence.
- Wheel: `python -m pip wheel --no-deps --no-build-isolation .` succeeded; installed built wheel into a clean venv without torch and listed three real traces successfully. No external packages needed for analyzer-only mode.

## Executed examples

Both tiny Transformer examples completed. Each recorded 12 module calls and 15 unique parameter tensors. The following gradient norms are observed on the embedding, not model-wide norms:

| Trace | CE gradient | Auxiliary gradient | Cosine |
|---|---:|---:|---:|
| healthy | 0.1468072163 | 0.0357098659 | 0.1088039017 |
| detached | unused (null) | unused (null) | undefined (null) |

Head gradient norms were identical between runs (CE 0.9781162712, auxiliary 0.3268446181). The controlled example deliberately detaches the intermediate representation; it confirms the expected loss of connectivity in this toy case, not universal detach detection.

Frozen toy model with external soft embedding: CE norm 0.1862506032, KD norm 0.0479932234, cosine -0.8512898070. Trace includes the external target and optimizer membership. These are synthetic example measurements and are not the user's BE research results. Samples in `examples/sample_traces` are the recorded outputs.

## Not yet verified

- Real MCP stdio protocol, Hermes/other client integration, tool naming and skill activation behavior; test provided in `tests/test_mcp.py`.
- macOS MPS, CUDA, Windows, Python 3.10/3.12 matrix; GitHub Actions provided but not executed here.
- Distributed, compiled, concurrent, exotic tensor or arbitrary model architectures (out of initial scope); memory/latency overhead and agent answer quality.
- Security under malicious concurrent directory mutation or hostile model code. Local private trace directory required. No cryptographic provenance.

## Review decisions

Separate optional torch and MCP dependencies; avoid arbitrary execution in MCP; bound output, record evidence instead of asserting causal root cause; no raw tensor values; no GPU timing claim; explicit gradient probe and snapshot stage. Proposed project/distribution names and MIT license should be reviewed before publication. No GitHub repository or PyPI release created.
