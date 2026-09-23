# 0018 Tool Selection: Bedrock prototype benchmark

Executable evidence for the [Tool Selection design](https://github.com/strands-agents/harness-sdk/pull/4008) (`team/designs/0018-tool-selection.md`). It compares sending the full tool catalog with per-invocation lexical selection against a real Bedrock model, and reports prompt tokens, cache reads and writes, normalized cost, and latency.

The selector here is a private prototype plugin keyed by agent, not the proposed public API. The design keys selection state by invocation and puts the policy on `ContextManager`; this script only proves the mechanism and the cache economics.

## Method

- `BedrockModel("us.anthropic.claude-haiku-4-5-20251001-v1:0")` in `us-west-2`, `context_manager="auto"` in both arms, 1-hour tool, system, and message cache points.
- Deterministic catalog of 12 domains × 10 actions (120 callable tools) with padded, stable descriptions; a 60-tool prefix for the smaller cells.
- The selected arm retains 24 tools per invocation, chosen by lexical overlap with the latest user text, and preserves registry order.
- Four tasks per cell: a repeated billing search (same selected set), a billing summary, and a shipping search (changed sets), against a short history (no seed) and a long history (16,000 neutral words, roughly 16k additional cached tokens).
- Each arm uses a fresh agent with the same catalog, seed history, prompts, and model configuration. Each invocation makes two model calls (tool request, then final answer).

## Results

Both arms invoked the expected tool on **16/16 tasks** across the four cells.

| Catalog / history | Full prompt tokens | Selected prompt tokens | Prompt Δ | Normalized cost Δ | Cache-write Δ | Latency Δ |
|---|---:|---:|---:|---:|---:|---:|
| 60 tools / short | 296.0k | 126.6k | -57.2% | **+3.3%** | +28.5% | -12.0% |
| 60 tools / +16k words | 427.8k | 258.4k | -39.6% | **+45.0%** | +81.1% | +17.4% |
| 120 tools / short | 544.2k | 119.9k | -78.0% | **-46.2%** | -33.4% | -32.6% |
| 120 tools / +16k words | 674.0k | 221.7k | -67.1% | **-20.5%** | -1.1% | -25.1% |

"Prompt tokens" is the provider-reported sum of ordinary input, cache-read, and cache-write tokens across the eight model calls in a cell. "Normalized cost" uses the published Claude Haiku 4.5 relative rates for the configured 1-hour cache: ordinary input 1×, cache read 0.1×, cache write 2×, output 5×. It is a comparable cost index, not an AWS invoice.

Cache behavior matched the provider documentation. The stable full catalog achieved an 87.4 to 87.5 percent cache-read ratio after its initial write. Repeating the same selected set produced a first-cycle cache read on the next invocation. Changing selected membership produced zero first-cycle cache reads and rewrote the selected tools plus the conversation prefix. Selection always reduced raw context volume, but at 60 tools the extra writes outweighed the savings; at 120 tools the removed definitions were large enough to win even with long history.

## Interpretation

The prototype establishes the mechanism and the cache break-even behavior, not general retrieval quality. Selection preserved tool correctness while reducing the visible catalog to 24 of 60 or 120 tools. Raw token reduction is not a sufficient success metric, because cache reads, writes, and provider pricing change the result. Tool count alone is not a safe activation threshold; tool-spec size, conversation size, cache warmth, and selection stability all matter.

Limitations: one model, provider, and region; synthetic padded descriptions; four deterministic tasks per cell; one run per cell; no statistical latency analysis. Follow-up benchmarks should use real MCP and local-tool catalogs, ambiguous intent, different cache TTLs, and multiple models.

## Reproduction

Run from `strands-py` with valid Bedrock credentials. The script never makes a live request unless `--run` is supplied.

```bash
# Dry-run selection only; no AWS calls
hatch run python ../prototypes/0018-tool-selection/bedrock_prototype.py

# One matrix cell (paid)
hatch run python ../prototypes/0018-tool-selection/bedrock_prototype.py \
  --run --mode both --catalog-size 60 --history-words 16000 \
  --scenario-limit 4 --output /tmp/tool-selection-60-long.json
```

`--catalog-size` selects the 60- or 120-tool cell, `--history-words` seeds the long-history cell, and `--output` writes per-call usage as JSON for independent analysis.
