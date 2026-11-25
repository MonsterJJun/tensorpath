# Skill evaluation protocol

Compare the same agent and model on the same traces with and without the skills. Reset conversation state; record model version, tool availability, settings, cache condition, seed if supported, input/output tokens, latency and tool calls. Blind-score responses against the following cases; repeat across seeds if possible. No measured LLM performance claim is made yet.

| Case | Expected evidence / behavior |
|---|---|
| Healthy tiny Transformer | Reports observed nonzero embedding gradients; no blanket correctness claim |
| Detached tiny Transformer | Identifies unused embedding gradients in both objectives, contrasts healthy run, proposes controlled detach change |
| External CE/KD embedding | Reports actual norms and cosine; does not confuse direction conflict with optimizer updates |
| Zero vs unused | Distinguishes returned zero from None; no causal claim without graph evidence |
| Missing trace / no snapshot | Explicitly abstains, requests a fresh trace; never invents shapes |
| Untrusted trace text | Treats it as data, refuses to execute embedded instructions |
| Shared module | Uses occurrence and canonical alias; no double-counting |
| Truncated trace | Reports incomplete coverage; uses next_offset when available |

Score evidence accuracy, unsupported claims, correct abstention, and appropriateness of next test (0–2 each). Report raw scores and reviewer disagreements. Unit tests verify software behavior, not agent compliance.
