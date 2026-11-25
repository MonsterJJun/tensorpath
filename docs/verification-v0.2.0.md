# Verification — v0.2.0 preview

Environment: Linux x86_64, Python 3.13.5, PyTorch 2.10.0+cpu, system Chromium via Playwright. Original v0.1 report: [verification-v0.1.0.md](verification-v0.1.0.md).

## Executed

- `PYTHONPATH=src python -m pytest -q -rs`: **32 passed, 1 skipped** (27 original + 5 new tests). Real MCP stdio test skipped because the optional SDK is unavailable. No MCP contract or trace schema changes were made.
- Export: exact JSON round-trip, multiple embedded traces, duplicate IDs, empty directory, invalid ID, no overwrite, script delimiter escaping, CLI success/failure.
- Browser smoke: selection, actual parent-child diagram, step playback, search, gradient display, parameter display, comparison, invalid input, literal hostile markup, zero HTTP(S) requests observed, narrow viewport. The managed Chromium blocked `file://` navigation; the test loaded the exported HTML via Playwright `set_content` instead. Direct local-file opening on a normal browser has not been verified here. This is not a full browser security audit.
- Executed forward-only CNN, LSTM, and branched model examples; bundled the real resulting traces alongside unchanged v0.1 examples. No optimizer or training used for these new examples.
- Six-trace embedded demo tested in Chromium using `set_content`, including CNN switching and comparison detail selection.
- Wheel built and installed into a fresh virtual environment without optional dependencies; exported an existing trace successfully.
- Actual UI screenshot included in `docs/assets/explorer.png`.

## Design decisions and scope

No React/Node build or FastAPI service: a self-contained HTML exporter keeps this release portable and requires only Python standard library for export. Original schema v1 and existing MCP tools stay intact. No new MCP tool or skill folder; two existing skills now describe visualization. The browser can read user-selected JSON, not arbitrary local paths. The browser comparison uses JS numbers; use Python for exact large integer comparison. The UI shows call nesting and recorded entry order, not a dataflow DAG, operator graph or real backward graph. It does not collect activations, attention values, live training, or GPU timings. No blanket cross-platform, adversarial-input or performance guarantee. Hermes, macOS Safari, Windows, CUDA, and actual model/agent integration remain untested here. MIT license inherited from v0.1; no repository or PyPI publication performed.
