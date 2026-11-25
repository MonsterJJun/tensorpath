# Verification — v0.2.1 preview

## Executed checks

- Linux x86_64, Python 3.13, PyTorch CPU. `python -m pytest -q -rs`: **44 passed, 1 skipped**. Includes v0.2.0 regression tests, static inventory, scoped report creation, forward-only capture, setup preview/idempotence/conflict/rollback/SDK preflight, and output symlink rejection.
- Setup tests use temporary Hermes homes and simulate SDK availability when testing actual config writes. No real Hermes installation was changed. A copy of the supplied Hermes YAML was previewed and updated in a temporary directory with simulated SDK availability: the original backup was byte-identical, and parsed configuration values outside the TensorPath entry were equal before/after. Comments are preserved by round-trip YAML where supported; formatting may change. Conflicting TensorPath entries are fully replaced only with explicit `--replace` and backup.
- Built a wheel and installed it into a fresh venv without optional dependencies; confirmed ten bundled skills, no torch import, and successful Explorer HTML export.
- Chromium browser regression: selection, call hierarchy, replay, search, gradients, comparison, invalid input, literal hostile markup, no HTTP(S) requests, mobile viewport. The managed browser blocked `file://`; generated HTML was loaded via `set_content` instead.
- Original schema v1 and existing six MCP tool signatures retained. Two new tool wrappers and their underlying functions were implemented; stdio round-trip test includes new endpoints.

## Not verified

- The MCP SDK is unavailable in this environment. Real stdio test is skipped; `doctor --protocol` is provided for local protocol verification. Actual Hermes integration, agent tool selection, approval behavior, skill compliance, and macOS/Windows/CUDA execution are not validated. CI is provided but not run on GitHub here.
- Static discovery is heuristic and bounded; no arbitrary project execution occurs in MCP. The recording skill asks the agent to propose a minimal capture and obtain approval, then use its host terminal if available. This is an instruction workflow, not a sandbox or a fully autonomous capture engine. Forward-only does not prevent custom model side effects or mutable buffers.
- No new operator-level dataflow, full autograd graph, MLX backend, or raw activation collection. The v0.2 Explorer is unchanged except for output-path hardening. No blanket security, performance or cross-platform guarantee.
- No PyPI publication, GitHub push, or installation on the user's Mac. MIT license inherited. Use private, stable project, trace and report directories; setup is backup-first and best-effort rollback, not a transaction against concurrent filesystem editors.

Historical checks: [v0.2.0](docs/verification-v0.2.0.md), [v0.1.0](docs/verification-v0.1.0.md).
