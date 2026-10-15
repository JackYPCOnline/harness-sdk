# 0021 cache-friendly tool management: Bedrock prototype

Evidence for design 0021 (`team/designs/0021-cache-friendly-tool-management.md`), Appendix C. It tests the Bedrock Converse stub path, where `BedrockModel` has no native deferred-tool support. A tools list fixed for the conversation is compared against sending every spec and against replacing the tools list per invocation, the way `Hide.drop` does.

`skeleton_prototype.py` places the pieces the way the design does:
- An `InvokeModelStage.Input` middleware renders a plan decided on the conversation's first model call.
- The plan lives in `agent.state`, so session managers persist it.
- Loaded tools are read back from the message history.
- A pinned `load_tools` tool returns definitions as ordinary tool results.
- An `ExecuteToolStage` guard answers a call to an unloaded stub with its definition.

It's a private prototype plugin, not the proposed public API.

## Method

- **Model:** Claude Haiku 4.5 (`us.anthropic.claude-haiku-4-5-20251001-v1:0`) on Converse in `us-west-2`, with `streaming=False`.
- **Caching:** 1-hour cache points on tools, system, and messages.
- **Catalog:** the Appendix A catalog of 12 domains × 10 actions (120 tools, plus a 60-tool prefix) with the same padded descriptions. Each action has a realistic input schema with nested objects, enums, formats, booleans, and arrays.
- **Arms:**
  - `full`: every spec on every call.
  - `selected`: a lexical top 24 per invocation that replaces the tools list (the Appendix A selector).
  - `skeleton --skeleton-mode schema` (open-schema stubs in the design): the stub has an open object schema and a pointer to `load_tools`, and names and first sentences are listed in the loader. This is the design's stub with the description deferred too.
  - `skeleton --skeleton-mode description` (summary stubs in the design): the first sentence of the description, with the real schema kept. Rejected in the design.
- **Cost protocol:** four invocations per cell, against short history and +16k words: the same tool twice, a related tool, then a new domain.
- **Accuracy protocol:** five invocations on the 120-tool catalog, each needing exact nested, enum, boolean, or array arguments.
- **Correctness:** an `ExecuteToolStage` observer checks every executed call against the real JSON schema (`jsonschema`, Draft 2020-12, with format checks) and the expected values.
- **Request capture:** every Converse request and response is captured through botocore events, so each call's cache counters and prefix stability can be checked.
- **Restore check:** two invocations with a `FileSessionManager`, then a new agent restored from that session.
- **Normalized cost:** Haiku 4.5 relative rates of input 1×, cache read 0.1×, 1-hour cache write 2×, and output 5×. It's an index, not an invoice.
- **Context manager:** `--context-manager none` in the tables, so only tool handling differs. `claude-haiku-4-5_cost_auto-context-manager.json` is a separate run with `auto`.

## Results

Cost protocol (normalized cost, Δ against `full`):

| Catalog / history | `full` | `selected` | Open-schema stubs | Summary stubs |
|---|---:|---:|---:|---:|
| 60 tools / short | 123.5k | +0.7% | −78.4% | −68.0% |
| 60 tools / +16k words | 170.2k | +37.3% | −55.1% | −51.1% |
| 120 tools / short | 241.4k | −34.9% | −84.3% | −69.7% |
| 120 tools / +16k words | 289.8k | +2.9% | −69.7% | −59.7% |

The summary-stub column ran separately, against its own `full` baseline (129.4k, 175.8k, 253.0k, 301.8k).

- **Correctness:** every arm called the expected tool with correct arguments on all 21 tasks (16 cost, 5 accuracy).
- **Model calls per cell:** 8 for `full`, `selected`, and summary stubs; 11 for open-schema stubs, including 3 loads. A loaded tool was reused without reloading, and the guard never fired.
- **Tools-list changes per cell:** 2–3 for `selected`, each a full prefix rewrite; 0 for both stub arms. No request in any arm edited an earlier message.
- **Session restore** (open-schema stubs, 120 tools): the plan was restored, not recomputed. The first request after restoring had the same tools list and system prompt, appended to its messages, and read 9,549 tokens from cache.
- **`auto` context manager:** the first call was sized from the full registry. At 120 tools and +16k words it summarized before the first call in every arm, including the stub arms.

## Limitations

- One run per cell, one model, Converse only. The Claude API and OpenAI Responses paths aren't tested.
- Neither stub arm is the design's final shape (full description, deferred schema). The design estimates that shape from the measured schema share.
- The descriptions are synthetic padding, and the prompts state the argument values.
- The live runs cost about $6 at Haiku list prices.

## Raw results

`results/` holds the JSON behind every number above: per-call usage and prefix-stability flags, per-invocation outcomes with the exact arguments sent, and per-arm summaries.

## Reproduction

The script makes no AWS calls unless `--run` is given. Run it from `strands-py` with this checkout installed, for example `uv venv && uv pip install -e .`.

```bash
# Dry run: tools-section size estimates
python ../prototypes/0021-tool-skeleton/skeleton_prototype.py

# Cost protocol, open-schema stubs
python ../prototypes/0021-tool-skeleton/skeleton_prototype.py --run --context-manager none --output /tmp/cost.json

# Cost protocol, summary stubs
python ../prototypes/0021-tool-skeleton/skeleton_prototype.py --run --context-manager none \
    --skeleton-mode description --arms full,skeleton

# Accuracy protocol
python ../prototypes/0021-tool-skeleton/skeleton_prototype.py --run --context-manager none \
    --scenarios accuracy --catalog-sizes 120 --history-words 0

# Session restore
python ../prototypes/0021-tool-skeleton/skeleton_prototype.py --run --context-manager none \
    --restore-check --catalog-sizes 120
```
