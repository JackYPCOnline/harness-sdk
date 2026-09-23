"""Bedrock benchmark prototype for dynamic tool selection and prompt caching.

This script exercises the proposed mechanism against a real Bedrock model while
using the SDK's existing ``context_manager="auto"`` facade. The selector is a
private prototype plugin, not the proposed public API. It filters the defensive
``InvokeModelContext.tool_specs`` copy once per outer agent invocation and
reapplies that selection throughout the invocation's tool loop.

The benchmark compares two arms over the same sequence of prompts:

* ``full``: every registered tool spec is sent on every model call.
* ``selected``: a lexical selector keeps a fixed subset per invocation.

The catalog contains 120 callable tools (12 domains x 10 actions) with padded,
stable descriptions. Repeated and related prompts reveal tool-set overlap,
provider cache reads/writes, and the cost of changing membership.

Run from ``strands-py`` so the local package is used:

    # No AWS calls; prints the catalog and planned selections.
    hatch run python ../prototypes/0018-tool-selection/bedrock_prototype.py

    # Paid Bedrock calls, selected arm only.
    hatch run python ../prototypes/0018-tool-selection/bedrock_prototype.py \
        --run --mode selected --scenario-limit 2 --profile YOUR_PROFILE

    # Paid comparison of full and selected arms.
    hatch run python ../prototypes/0018-tool-selection/bedrock_prototype.py \
        --run --mode both --profile YOUR_PROFILE --output /tmp/tool-selection.json

The live run uses real AWS credentials and incurs Bedrock charges. The script
never runs a live request unless ``--run`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal
from weakref import WeakKeyDictionary

import boto3
from strands import Agent, Plugin, tool
from strands._middleware.stages import InvokeModelContext, InvokeModelStage
from strands.agent import AgentResult
from strands.hooks import BeforeInvocationEvent
from strands.models import BedrockModel, CacheConfig, CacheToolsConfig
from strands.types.content import Message, Messages
from strands.types.tools import AgentTool, ToolSpec, ToolUse

Mode = Literal["full", "selected"]

_DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_DEFAULT_REGION = "us-west-2"
_DEFAULT_SELECTION_LIMIT = 24
_DEFAULT_DESCRIPTION_REPEAT = 12
_CHARS_PER_TOKEN_ESTIMATE = 4
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

_DOMAINS: dict[str, tuple[str, ...]] = {
    "billing": ("invoice", "payment", "charge", "refund", "subscription", "account"),
    "shipping": ("shipment", "tracking", "carrier", "delivery", "route", "package"),
    "inventory": ("stock", "warehouse", "sku", "quantity", "restock", "allocation"),
    "security": ("threat", "incident", "vulnerability", "audit", "access", "risk"),
    "identity": ("user", "group", "role", "permission", "authentication", "profile"),
    "analytics": ("metric", "dashboard", "trend", "cohort", "funnel", "report"),
    "support": ("ticket", "case", "customer", "resolution", "escalation", "service"),
    "procurement": ("vendor", "purchase", "order", "contract", "supplier", "approval"),
    "compliance": ("policy", "control", "evidence", "regulation", "review", "attestation"),
    "monitoring": ("alarm", "log", "trace", "health", "availability", "performance"),
    "finance": ("budget", "forecast", "ledger", "expense", "revenue", "allocation"),
    "communications": ("message", "channel", "email", "notification", "campaign", "recipient"),
}

_ACTIONS: dict[str, tuple[str, ...]] = {
    "search": ("find", "query", "lookup", "discover"),
    "list": ("enumerate", "browse", "show", "inventory"),
    "get": ("fetch", "read", "retrieve", "inspect"),
    "summarize": ("summary", "aggregate", "digest", "overview"),
    "compare": ("difference", "contrast", "baseline", "change"),
    "validate": ("verify", "check", "confirm", "test"),
    "export": ("download", "extract", "archive", "serialize"),
    "create": ("add", "open", "provision", "register"),
    "update": ("edit", "modify", "change", "patch"),
    "diagnose": ("debug", "investigate", "troubleshoot", "analyze"),
}


@dataclass(frozen=True)
class Scenario:
    """One outer agent invocation in the protocol."""

    name: str
    prompt: str
    expected_tool: str


_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "billing_search_a",
        "Search billing records for invoice INV-2026-0042. Call exactly one relevant tool.",
        "billing_search",
    ),
    Scenario(
        "billing_search_b",
        "Search billing records for invoice INV-2026-0099. Call exactly one relevant tool.",
        "billing_search",
    ),
    Scenario(
        "billing_summary",
        "Summarize billing charges for account ACCT-17. Call exactly one relevant tool.",
        "billing_summarize",
    ),
    Scenario(
        "shipping_search_a",
        "Search shipping records for package PKG-431. Call exactly one relevant tool.",
        "shipping_search",
    ),
    Scenario(
        "shipping_search_b",
        "Search shipping records for package PKG-982. Call exactly one relevant tool.",
        "shipping_search",
    ),
    Scenario(
        "security_diagnose",
        "Diagnose the security incident SEC-77. Call exactly one relevant tool.",
        "security_diagnose",
    ),
    Scenario(
        "identity_diagnose",
        "Diagnose the identity access incident IAM-14. Call exactly one relevant tool.",
        "identity_diagnose",
    ),
)

_SYSTEM_PROMPT = """You are running a tool-selection benchmark.
For every user request, call exactly one available tool whose domain and action
best match the request. After the tool returns, answer in one short sentence and
do not call another tool. Never invent a tool name. The tool outputs are
synthetic benchmark data and require no external side effects.
"""


@dataclass(frozen=True)
class SelectionOutcome:
    """Result of applying the prototype lexical selector."""

    selected_names: tuple[str, ...]
    ranked: tuple[tuple[str, int], ...]
    failed_open: bool


@dataclass
class _InvocationSelectionState:
    """Private selector state for one Agent's current outer invocation."""

    invocation_number: int = 0
    query: str = ""
    selected_names: tuple[str, ...] | None = None
    catalog_count: int = 0
    incoming_count: int = 0
    visible_count: int = 0
    failed_open: bool = False
    ranked: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class UsageSnapshot:
    """Token counters for one invocation or one cycle."""

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_write_input_tokens: int

    @property
    def prompt_tokens(self) -> int:
        """Total prompt tokens including uncached, cache-read, and cache-write tokens."""
        return self.input_tokens + self.cache_read_input_tokens + self.cache_write_input_tokens


@dataclass
class Observation:
    """One scenario result in one benchmark arm."""

    mode: Mode
    scenario: str
    prompt: str
    expected_tool: str
    invoked_tools: list[str]
    expected_tool_invoked: bool
    catalog_count: int
    selected_count: int
    visible_count: int
    selected_names: list[str]
    selection_overlap: float | None
    failed_open: bool
    usage: UsageSnapshot
    cycle_usage: list[UsageSnapshot]
    cycles: int
    elapsed_seconds: float
    response: str


@dataclass
class ArmSummary:
    """Aggregate metrics for one benchmark arm."""

    mode: Mode
    invocations: int
    total_cycles: int
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_input_tokens: int
    total_cache_write_input_tokens: int
    total_prompt_tokens: int
    expected_tool_successes: int
    average_selected_tools: float
    elapsed_seconds: float


@dataclass
class ProtocolResult:
    """Serializable benchmark output."""

    salt: str
    model_id: str
    region: str
    cache_ttl: str | None
    catalog_size: int
    selection_limit: int
    description_repeat: int
    history_words: int
    observations: list[Observation] = field(default_factory=list)
    summaries: list[ArmSummary] = field(default_factory=list)


def _tokenize(text: str) -> set[str]:
    """Return normalized lexical terms."""
    return set(_TOKEN_PATTERN.findall(text.casefold()))


def _schema_text(spec: ToolSpec) -> str:
    """Render searchable input-property names and descriptions from a ToolSpec."""
    schema = spec.get("inputSchema", {}).get("json", {})
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    if not isinstance(properties, dict):
        return ""

    parts: list[str] = []
    for property_name, property_schema in properties.items():
        parts.append(str(property_name))
        if isinstance(property_schema, dict):
            description = property_schema.get("description")
            if description:
                parts.append(str(description))
    return " ".join(parts)


def _candidate_text(spec: ToolSpec) -> str:
    """Render the fields used by the lexical selector."""
    return f"{spec['name'].replace('_', ' ')} {spec['description']} {_schema_text(spec)}"


def _latest_user_text(messages: Messages) -> str:
    """Extract top-level text from the latest user input, skipping tool-result-only turns."""
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        text = "\n".join(block["text"] for block in message["content"] if "text" in block).strip()
        if text:
            return text
    return ""


def _lexical_select(specs: Sequence[ToolSpec], query: str, limit: int) -> SelectionOutcome:
    """Rank specs lexically and return a deterministic catalog-ordered subset."""
    if len(specs) <= limit:
        names = tuple(spec["name"] for spec in specs)
        return SelectionOutcome(names, tuple((name, 0) for name in names), failed_open=False)

    query_terms = _tokenize(query)
    scored: list[tuple[int, int, str]] = []
    for index, spec in enumerate(specs):
        name_terms = _tokenize(spec["name"].replace("_", " "))
        candidate_terms = _tokenize(_candidate_text(spec))
        score = 4 * len(query_terms & name_terms) + len(query_terms & candidate_terms)
        scored.append((score, index, spec["name"]))

    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    if not ranked or ranked[0][0] <= 0:
        names = tuple(spec["name"] for spec in specs)
        return SelectionOutcome(names, tuple((name, score) for score, _, name in ranked), failed_open=True)

    selected_set = {name for _, _, name in ranked[:limit]}
    # Providers receive catalog order, never score order, to avoid reorder-only cache misses.
    selected_names = tuple(spec["name"] for spec in specs if spec["name"] in selected_set)
    ranked_names = tuple((name, score) for score, _, name in ranked)
    return SelectionOutcome(selected_names, ranked_names, failed_open=False)


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float | None:
    """Return set overlap, or None when there is no previous set."""
    if not left and not right:
        return 1.0
    if not right:
        return None
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _estimate_spec_tokens(specs: Sequence[ToolSpec]) -> int:
    """Estimate serialized tool tokens without a provider call."""
    chars = sum(len(json.dumps(spec, sort_keys=True, separators=(",", ":"))) for spec in specs)
    return chars // _CHARS_PER_TOKEN_ESTIMATE


def _extract_tool_uses(messages: Sequence[Message]) -> list[ToolUse]:
    """Collect tool-use blocks in message order."""
    return [block["toolUse"] for message in messages for block in message["content"] if "toolUse" in block]


def _usage_snapshot(usage: dict[str, Any]) -> UsageSnapshot:
    """Copy SDK usage counters before another invocation mutates shared metrics."""
    return UsageSnapshot(
        input_tokens=int(usage.get("inputTokens", 0)),
        output_tokens=int(usage.get("outputTokens", 0)),
        cache_read_input_tokens=int(usage.get("cacheReadInputTokens", 0)),
        cache_write_input_tokens=int(usage.get("cacheWriteInputTokens", 0)),
    )


def _tool_call_counts(agent: Agent) -> dict[str, int]:
    """Snapshot cumulative tool call counts."""
    return {name: metric.call_count for name, metric in agent.event_loop_metrics.tool_metrics.items()}


def _tool_call_deltas(agent: Agent, before: dict[str, int]) -> dict[str, int]:
    """Return positive tool call deltas since a snapshot."""
    return {
        name: delta
        for name, metric in agent.event_loop_metrics.tool_metrics.items()
        if (delta := metric.call_count - before.get(name, 0)) > 0
    }


def _make_catalog_tool(
    domain: str,
    action: str,
    domain_terms: Sequence[str],
    action_terms: Sequence[str],
    *,
    description_repeat: int,
    salt: str,
) -> AgentTool:
    """Create one stable, callable benchmark tool."""
    tool_name = f"{domain}_{action}"
    semantic = (
        f"Use {tool_name} for the {domain} domain and the {action} action. "
        f"Domain concepts: {', '.join(domain_terms)}. "
        f"Action concepts: {', '.join(action_terms)}. "
        "Choose this tool only when both its domain and operation match the request. "
    )
    padding = (
        f"Stable benchmark metadata {salt}: this {domain} capability performs {action} operations "
        f"over {', '.join(domain_terms)} and returns concise synthetic reference data. "
    )
    description = semantic + padding * description_repeat
    payload = {
        "tool": tool_name,
        "domain": domain,
        "action": action,
        "status": "synthetic_success",
    }

    @tool(name=tool_name, description=description)
    def catalog_tool(query: str) -> str:
        return json.dumps({**payload, "query": query}, sort_keys=True)

    return catalog_tool


def _build_catalog(description_repeat: int, salt: str, catalog_size: int) -> list[AgentTool]:
    """Build a deterministic, interleaved prefix of the 120-tool catalog."""
    # Action-major ordering ensures one domain's selected tools are not simply a prefix
    # of the full catalog, making full-vs-selected cache entries easier to distinguish.
    full_catalog = [
        _make_catalog_tool(
            domain,
            action,
            _DOMAINS[domain],
            _ACTIONS[action],
            description_repeat=description_repeat,
            salt=salt,
        )
        for action in _ACTIONS
        for domain in _DOMAINS
    ]
    return full_catalog[:catalog_size]


def _build_seed_messages(history_words: int, salt: str) -> Messages:
    """Build deterministic alternating history shared by both benchmark arms."""
    if history_words == 0:
        return []

    vocabulary = (
        "stable context history project customer requirement decision observation analysis evidence "
        "architecture workflow service account document review outcome reference discussion response"
    ).split()
    message_count = min(24, max(2, history_words // 250))
    if message_count % 2:
        message_count += 1

    base_count, remainder = divmod(history_words, message_count)
    messages: Messages = []
    cursor = 0
    for message_index in range(message_count):
        count = base_count + (1 if message_index < remainder else 0)
        words = [vocabulary[(cursor + offset) % len(vocabulary)] for offset in range(count)]
        cursor += count
        text = f"Synthetic stable history {salt} segment {message_index}. " + " ".join(words)
        messages.append({"role": "user" if message_index % 2 == 0 else "assistant", "content": [{"text": text}]})
    return messages


class InvocationLexicalToolFilter(Plugin):
    """Prototype-only filter: select once, reapply throughout the invocation."""

    name = "prototype:invocation-lexical-tool-filter"

    def __init__(self, catalog_names: Sequence[str], limit: int) -> None:
        self._catalog_names = frozenset(catalog_names)
        self._limit = limit
        self._states: WeakKeyDictionary[Agent, _InvocationSelectionState] = WeakKeyDictionary()
        super().__init__()

    def _before_invocation(self, event: BeforeInvocationEvent) -> None:
        state = self._states.setdefault(event.agent, _InvocationSelectionState())
        state.invocation_number += 1
        state.query = _latest_user_text(event.messages or [])
        state.selected_names = None
        state.catalog_count = 0
        state.incoming_count = 0
        state.visible_count = 0
        state.failed_open = False
        state.ranked = ()

    def init_agent(self, agent: Agent) -> None:
        self._states.setdefault(agent, _InvocationSelectionState())
        agent.add_hook(self._before_invocation, BeforeInvocationEvent)
        agent._middleware_registry.add_middleware(InvokeModelStage.Input, self._filter_tool_specs)

    def decision(self, agent: Agent) -> _InvocationSelectionState:
        """Return the latest decision for reporting."""
        return self._states[agent]

    def preview(self, query: str, specs: Sequence[ToolSpec]) -> SelectionOutcome:
        """Run the same selection logic without an Agent invocation."""
        catalog_specs = [spec for spec in specs if spec["name"] in self._catalog_names]
        return _lexical_select(catalog_specs, query, self._limit)

    def _filter_tool_specs(self, context: InvokeModelContext) -> InvokeModelContext:
        # Explicit tool choice and forced structured-output requests must pass through.
        if context.tool_choice is not None:
            return context

        state = self._states.setdefault(context.agent, _InvocationSelectionState())
        catalog_specs = [spec for spec in context.tool_specs if spec["name"] in self._catalog_names]
        state.catalog_count = len(catalog_specs)
        state.incoming_count = len(context.tool_specs)

        if state.selected_names is None:
            query = state.query or _latest_user_text(context.messages)
            outcome = _lexical_select(catalog_specs, query, self._limit)
            state.query = query
            state.selected_names = outcome.selected_names
            state.failed_open = outcome.failed_open
            state.ranked = outcome.ranked

        selected_set = set(state.selected_names)
        # Preserve all non-benchmark tools, notably ContextOffloader's retrieval tool.
        filtered = [
            spec
            for spec in context.tool_specs
            if spec["name"] not in self._catalog_names or spec["name"] in selected_set
        ]
        state.visible_count = len(filtered)
        return replace(context, tool_specs=filtered)


def _build_model(args: argparse.Namespace) -> BedrockModel:
    """Build a real Bedrock model with aligned tool/system/message cache TTLs."""
    boto_session = (
        boto3.Session(profile_name=args.profile, region_name=args.region) if args.profile is not None else None
    )
    cache_kwargs: dict[str, Any] = {}
    if args.cache_ttl != "none":
        cache_kwargs = {
            "cache_tools": CacheToolsConfig(type="default", ttl=args.cache_ttl),
            "cache_config": CacheConfig(
                strategy="auto",
                ttl=args.cache_ttl,
                system_prompt_ttl=args.cache_ttl,
            ),
        }

    return BedrockModel(
        boto_session=boto_session,
        region_name=args.region,
        model_id=args.model_id,
        max_tokens=args.max_tokens,
        streaming=False,
        context_window_limit=200_000,
        **cache_kwargs,
    )


def _build_agent(
    mode: Mode,
    args: argparse.Namespace,
    salt: str,
) -> tuple[Agent, InvocationLexicalToolFilter | None, tuple[str, ...]]:
    """Build one isolated benchmark arm."""
    catalog_tools = _build_catalog(args.description_repeat, salt, args.catalog_size)
    catalog_names = tuple(tool_instance.tool_name for tool_instance in catalog_tools)
    selector = InvocationLexicalToolFilter(catalog_names, args.selection_limit) if mode == "selected" else None
    model = _build_model(args)
    agent_tools: list[Any] = list(catalog_tools)
    seed_messages = _build_seed_messages(args.history_words, salt)
    agent = Agent(
        model=model,
        messages=seed_messages,
        tools=agent_tools,
        system_prompt=_SYSTEM_PROMPT,
        context_manager="auto",
        plugins=[selector] if selector is not None else None,
        load_tools_from_directory=False,
        callback_handler=None,
    )
    return agent, selector, catalog_names


def _observe_invocation(
    mode: Mode,
    scenario: Scenario,
    agent: Agent,
    selector: InvocationLexicalToolFilter | None,
    catalog_names: tuple[str, ...],
    previous_selected: Sequence[str],
) -> Observation:
    """Run and snapshot one paid invocation."""
    message_start = len(agent.messages)
    calls_before = _tool_call_counts(agent)
    started = time.monotonic()
    result: AgentResult = agent(scenario.prompt)
    elapsed = time.monotonic() - started

    invocation = result.metrics.latest_agent_invocation
    if invocation is None:
        raise RuntimeError("agent returned no invocation metrics")

    usage = _usage_snapshot(dict(invocation.usage))
    cycle_usage = [_usage_snapshot(dict(cycle.usage)) for cycle in invocation.cycles]
    new_messages = agent.messages[message_start:]
    message_tool_uses = _extract_tool_uses(new_messages)
    message_invoked_tools = [tool_use["name"] for tool_use in message_tool_uses]
    call_deltas = _tool_call_deltas(agent, calls_before)
    metric_invoked_tools = [name for name, count in call_deltas.items() for _ in range(count)]
    invoked_tools = message_invoked_tools or metric_invoked_tools
    if message_invoked_tools and set(call_deltas) != set(message_invoked_tools):
        print(f"  warning: message tool uses {message_invoked_tools} differ from metric deltas {call_deltas}")

    if selector is None:
        selected_names = catalog_names
        catalog_count = len(catalog_names)
        visible_count = len(agent.tool_registry.get_all_tool_specs())
        failed_open = False
    else:
        decision = selector.decision(agent)
        selected_names = decision.selected_names or ()
        catalog_count = decision.catalog_count
        visible_count = decision.visible_count
        failed_open = decision.failed_open

    return Observation(
        mode=mode,
        scenario=scenario.name,
        prompt=scenario.prompt,
        expected_tool=scenario.expected_tool,
        invoked_tools=invoked_tools,
        expected_tool_invoked=scenario.expected_tool in invoked_tools,
        catalog_count=catalog_count,
        selected_count=len(selected_names),
        visible_count=visible_count,
        selected_names=list(selected_names),
        selection_overlap=_jaccard(selected_names, previous_selected),
        failed_open=failed_open,
        usage=usage,
        cycle_usage=cycle_usage,
        cycles=len(invocation.cycles),
        elapsed_seconds=elapsed,
        response=str(result).strip(),
    )


def _summarize_arm(mode: Mode, observations: Sequence[Observation], elapsed: float) -> ArmSummary:
    """Aggregate one arm."""
    return ArmSummary(
        mode=mode,
        invocations=len(observations),
        total_cycles=sum(observation.cycles for observation in observations),
        total_input_tokens=sum(observation.usage.input_tokens for observation in observations),
        total_output_tokens=sum(observation.usage.output_tokens for observation in observations),
        total_cache_read_input_tokens=sum(observation.usage.cache_read_input_tokens for observation in observations),
        total_cache_write_input_tokens=sum(observation.usage.cache_write_input_tokens for observation in observations),
        total_prompt_tokens=sum(observation.usage.prompt_tokens for observation in observations),
        expected_tool_successes=sum(observation.expected_tool_invoked for observation in observations),
        average_selected_tools=(
            sum(observation.selected_count for observation in observations) / len(observations) if observations else 0.0
        ),
        elapsed_seconds=elapsed,
    )


def _print_observation(observation: Observation) -> None:
    """Print one readable invocation report."""
    overlap = "n/a" if observation.selection_overlap is None else f"{observation.selection_overlap:.2f}"
    print(f"\n[{observation.mode.upper()}] {observation.scenario}")
    print(
        f"  tools: selected={observation.selected_count}/{observation.catalog_count}, "
        f"visible_to_model={observation.visible_count}, overlap_with_previous={overlap}, "
        f"fail_open={observation.failed_open}"
    )
    print(
        f"  invoked={observation.invoked_tools or ['<none>']}, "
        f"expected={observation.expected_tool}, correct={observation.expected_tool_invoked}"
    )
    print(
        f"  usage: input={observation.usage.input_tokens}, "
        f"cache_read={observation.usage.cache_read_input_tokens}, "
        f"cache_write={observation.usage.cache_write_input_tokens}, "
        f"prompt_total={observation.usage.prompt_tokens}, "
        f"output={observation.usage.output_tokens}, cycles={observation.cycles}, "
        f"elapsed={observation.elapsed_seconds:.2f}s"
    )
    for cycle_index, cycle in enumerate(observation.cycle_usage, start=1):
        print(
            f"    cycle {cycle_index}: input={cycle.input_tokens}, read={cycle.cache_read_input_tokens}, "
            f"write={cycle.cache_write_input_tokens}, prompt_total={cycle.prompt_tokens}"
        )
    print(f"  response: {observation.response[:180]}")


def _print_summary(summary: ArmSummary) -> None:
    """Print aggregate arm metrics."""
    cache_ratio = (
        summary.total_cache_read_input_tokens / summary.total_prompt_tokens if summary.total_prompt_tokens else 0.0
    )
    print(f"\n=== {summary.mode.upper()} SUMMARY ===")
    print(
        f"invocations={summary.invocations}, cycles={summary.total_cycles}, "
        f"expected_tools={summary.expected_tool_successes}/{summary.invocations}, "
        f"avg_selected={summary.average_selected_tools:.1f}"
    )
    print(
        f"input={summary.total_input_tokens}, cache_read={summary.total_cache_read_input_tokens}, "
        f"cache_write={summary.total_cache_write_input_tokens}, "
        f"prompt_total={summary.total_prompt_tokens}, output={summary.total_output_tokens}, "
        f"cache_read_ratio={cache_ratio:.3f}, elapsed={summary.elapsed_seconds:.2f}s"
    )


def _dry_run(args: argparse.Namespace, salt: str) -> ProtocolResult:
    """Print the deterministic selection plan without AWS calls."""
    catalog = _build_catalog(args.description_repeat, salt, args.catalog_size)
    specs = [tool_instance.tool_spec for tool_instance in catalog]
    names = tuple(spec["name"] for spec in specs)
    selector = InvocationLexicalToolFilter(names, args.selection_limit)
    full_estimate = _estimate_spec_tokens(specs)
    print("DRY RUN — no AWS calls were made")
    print(f"catalog={len(specs)} tools, approximate full spec tokens={full_estimate:,}, salt={salt}")

    previous: Sequence[str] = ()
    scenarios = _SCENARIOS[: args.scenario_limit] if args.scenario_limit is not None else _SCENARIOS
    for scenario in scenarios:
        outcome = selector.preview(scenario.prompt, specs)
        selected_specs = [spec for spec in specs if spec["name"] in set(outcome.selected_names)]
        overlap = _jaccard(outcome.selected_names, previous)
        expected_present = scenario.expected_tool in outcome.selected_names
        print(
            f"{scenario.name:22} selected={len(outcome.selected_names):3}/{len(specs)}, "
            f"approx_tokens={_estimate_spec_tokens(selected_specs):6,}, "
            f"overlap={'n/a' if overlap is None else f'{overlap:.2f}'}, "
            f"expected_present={expected_present}, fail_open={outcome.failed_open}"
        )
        print(f"  {', '.join(outcome.selected_names)}")
        if not expected_present:
            raise RuntimeError(
                f"dry-run selection dropped expected tool {scenario.expected_tool!r} for {scenario.name!r}"
            )
        previous = outcome.selected_names

    print("\nAdd --run to execute paid Bedrock calls. Use --mode selected first to limit cost.")
    return ProtocolResult(
        salt=salt,
        model_id=args.model_id,
        region=args.region,
        cache_ttl=None if args.cache_ttl == "none" else args.cache_ttl,
        catalog_size=len(specs),
        selection_limit=args.selection_limit,
        description_repeat=args.description_repeat,
        history_words=args.history_words,
    )


def _run_live(args: argparse.Namespace, salt: str) -> ProtocolResult:
    """Execute requested paid benchmark arms."""
    modes: tuple[Mode, ...]
    if args.mode == "both":
        modes = ("full", "selected")
    else:
        modes = (args.mode,)

    print("LIVE BEDROCK RUN — this incurs AWS charges")
    print(
        f"model={args.model_id}, region={args.region}, profile={args.profile or '<default chain>'}, "
        f"cache_ttl={args.cache_ttl}, catalog_size={args.catalog_size}, history_words={args.history_words}, "
        f"salt={salt}"
    )
    print(
        "Cache counters aggregate tool, system, and message cache usage; compare arms and per-cycle patterns "
        "rather than attributing every read to the tool block."
    )
    protocol = ProtocolResult(
        salt=salt,
        model_id=args.model_id,
        region=args.region,
        cache_ttl=None if args.cache_ttl == "none" else args.cache_ttl,
        catalog_size=args.catalog_size,
        selection_limit=args.selection_limit,
        description_repeat=args.description_repeat,
        history_words=args.history_words,
    )

    for mode in modes:
        print(f"\nBuilding isolated {mode!r} arm...")
        agent, selector, catalog_names = _build_agent(mode, args, salt)
        arm_started = time.monotonic()
        arm_observations: list[Observation] = []
        previous_selected: Sequence[str] = ()
        scenarios = _SCENARIOS[: args.scenario_limit] if args.scenario_limit is not None else _SCENARIOS
        missing_expected = [
            scenario.expected_tool for scenario in scenarios if scenario.expected_tool not in catalog_names
        ]
        if missing_expected:
            raise ValueError(
                f"catalog size {args.catalog_size} omits expected scenario tools: {sorted(set(missing_expected))}"
            )

        for scenario in scenarios:
            observation = _observe_invocation(
                mode,
                scenario,
                agent,
                selector,
                catalog_names,
                previous_selected,
            )
            arm_observations.append(observation)
            protocol.observations.append(observation)
            _print_observation(observation)
            previous_selected = observation.selected_names

        summary = _summarize_arm(mode, arm_observations, time.monotonic() - arm_started)
        protocol.summaries.append(summary)
        _print_summary(summary)

    if len(protocol.summaries) == 2:
        full = next(summary for summary in protocol.summaries if summary.mode == "full")
        selected = next(summary for summary in protocol.summaries if summary.mode == "selected")
        print("\n=== SELECTED VS FULL ===")
        print(f"prompt token delta: {selected.total_prompt_tokens - full.total_prompt_tokens:+,}")
        print(f"cache read delta: {selected.total_cache_read_input_tokens - full.total_cache_read_input_tokens:+,}")
        print(f"elapsed delta: {selected.elapsed_seconds - full.elapsed_seconds:+.2f}s")

    return protocol


def _write_output(path: str | None, result: ProtocolResult) -> None:
    """Write JSON output when requested."""
    if path is None:
        return
    output_path = Path(path).expanduser()
    output_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print(f"\nWrote {output_path}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute paid Bedrock calls; default is dry-run")
    parser.add_argument("--mode", choices=("full", "selected", "both"), default="both")
    parser.add_argument("--profile", help="AWS profile name; omitted uses the default credential chain")
    parser.add_argument("--region", default=_DEFAULT_REGION)
    parser.add_argument("--model-id", default=_DEFAULT_MODEL_ID)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--cache-ttl", choices=("5m", "1h", "none"), default="1h")
    parser.add_argument("--selection-limit", type=int, default=_DEFAULT_SELECTION_LIMIT)
    parser.add_argument(
        "--catalog-size",
        type=int,
        default=len(_DOMAINS) * len(_ACTIONS),
        help="Use the first N tools from the deterministic 120-tool catalog",
    )
    parser.add_argument(
        "--history-words",
        type=int,
        default=0,
        help="Seed both arms with this many neutral history words before the scenarios",
    )
    parser.add_argument("--description-repeat", type=int, default=_DEFAULT_DESCRIPTION_REPEAT)
    parser.add_argument("--scenario-limit", type=int, help="Run only the first N scenarios (2 is a cache smoke test)")
    parser.add_argument("--salt", help="Stable cache salt; default generates a fresh value per run")
    parser.add_argument("--output", help="Optional JSON result path")
    args = parser.parse_args()
    if args.selection_limit <= 0:
        parser.error("--selection-limit must be positive")
    max_catalog_size = len(_DOMAINS) * len(_ACTIONS)
    if not 1 <= args.catalog_size <= max_catalog_size:
        parser.error(f"--catalog-size must be between 1 and {max_catalog_size}")
    if args.selection_limit > args.catalog_size:
        parser.error("--selection-limit cannot exceed --catalog-size")
    if args.history_words < 0:
        parser.error("--history-words must be non-negative")
    if args.description_repeat < 0:
        parser.error("--description-repeat must be non-negative")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.scenario_limit is not None and args.scenario_limit <= 0:
        parser.error("--scenario-limit must be positive")
    return args


def main() -> None:
    """Run the dry plan or paid benchmark."""
    args = _parse_args()
    salt = args.salt or uuid.uuid4().hex[:8]
    result = _run_live(args, salt) if args.run else _dry_run(args, salt)
    _write_output(args.output, result)


if __name__ == "__main__":
    main()
