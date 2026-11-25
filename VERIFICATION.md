# Verification — v0.2.1 preview

## Executed checks

- Linux x86_64, Python 3.13, PyTorch CPU. `python -m pytest -q -rs`: **44 passed, 1 skipped**. Includes v0.2.0 regression tests, static inventory, scoped report creation, forward-only capture, setup preview/idempotence/conflict/rollback/SDK preflight, and output symlink rejection.
- Setup tests use temporary Hermes homes and simulate SDK availability when testing actual config writes. No real Hermes installation was changed. A copy of the supplied Hermes YAML was previewed and updated in a temporary directory with simulated SDK availability: the original backup was byte-identical, and parsed configuration values outside the TensorPath entry were equal before/after. Comments are preserved by round-trip YAML where supported; formatting may change. Conflicting TensorPath entries are fully replaced only with explicit `--replace` and backup.
- Built a wheel and installed it into a fresh venv without optional dependencies; confirmed ten bundled skills, no torch import, and successful Explorer HTML export.
- Chromium browser regression: selection, call hierarchy, replay, search, gradients, comparison, invalid input, literal hostile markup, no HTTP(S) requests, mobile viewport. The managed browser blocked `file://`; generated HTML was loaded via `set_content` instead.
- Original schema v1 and existing six MCP tool signatures retained. Two new tool wrappers and their underlying functions were implemented; stdio round-trip test includes new endpoints.

## macOS Integration Verification

Date: 2026-11-25

Environment:
- MacBook Air M5, 32GB unified memory
- macOS / Apple Silicon
- Hermes Agent
- LM Studio
- Shared Python virtual environment

Results:
- Pytest: 45 tests passed
- MCP stdio protocol: passed
- MCP tools: 8 discovered
- Sample traces: 6 available
- Hermes skill activation: user-verified
- Agent tool calling: user-verified
- Interactive HTML generation: user-verified

Remaining limitations:
- Large-scale model compatibility
- Distributed training
- Operator-level dataflow tracing
- Quantitative agent performance evaluation


Historical checks: [v0.2.0](docs/verification-v0.2.0.md), [v0.1.0](docs/verification-v0.1.0.md).
