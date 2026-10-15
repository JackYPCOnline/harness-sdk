"""Bedrock prototype for design 0021: cache-friendly progressive tool disclosure.

Compares three arms over the same prompts against a real Bedrock model through the
Converse API (the path ``BedrockModel`` uses):

* ``full``: every catalog tool spec is sent on every model call. Baseline.
* ``selected``: a lexical selector replaces the tools list once per invocation. This is
  the replacement approach measured in Appendix A of design 0021 (the shape of ``Hide.drop``).
* ``skeleton``: the 0021 proposal's Bedrock Converse stub path. The tools list is decided on the
  conversation's first model call and never changes afterwards. Catalog tools are sent as
  stubs (real name, a pointer to ``load_tools``, open object schema). A pinned ``load_tools``
  tool carries the index and returns full definitions as ordinary tool results, so loads are
  appended to the conversation. An ``ExecuteToolStage`` guard stops a call to a tool that
  has not been loaded and returns its definition instead.

The skeleton arm follows the design's placement: an ``InvokeModelStage.Input`` middleware
renders the plan, the plan lives in ``agent.state`` (persisted by session managers), what is
loaded is read from message history, and the guard is ``ExecuteToolStage`` wrap middleware.
In the proposal the strategy only marks specs as deferred and the model layer renders stubs;
this prototype renders the Converse stubs directly because ``BedrockModel`` uses Converse.

Every Converse request and response is captured through botocore events, so the report
shows per-call cache reads and writes and whether the tools list, the system prompt, and
the message history stayed append-only relative to the previous request.

Run from ``strands-py`` with an environment that has this checkout installed:

    # No AWS calls: prints the catalog and tools-section size estimates.
    python ../prototypes/0021-tool-skeleton/skeleton_prototype.py

    # Paid Bedrock calls: the four Appendix A cells, all three arms.
    python ../prototypes/0021-tool-skeleton/skeleton_prototype.py --run --output /tmp/0021-cost.json

    # Paid Bedrock calls: argument accuracy on the 120-tool catalog.
    python ../prototypes/0021-tool-skeleton/skeleton_prototype.py --run --scenarios accuracy \
        --catalog-sizes 120 --history-words 0

    # Paid Bedrock calls: session save and restore with the skeleton arm.
    python ../prototypes/0021-tool-skeleton/skeleton_prototype.py --run --restore-check

The live run uses real AWS credentials and incurs Bedrock charges.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal
from weakref import WeakKeyDictionary

import boto3
import jsonschema
from strands import Agent, tool
from strands._middleware.stages import ExecuteToolContext, ExecuteToolStage, InvokeModelContext, InvokeModelStage
from strands.agent.conversation_manager import NullConversationManager
from strands.hooks import BeforeInvocationEvent
from strands.models import BedrockModel, CacheConfig
from strands.plugins import Plugin
from strands.session import FileSessionManager
from strands.tools.tools import PythonAgentTool
from strands.types._events import ToolResultEvent
from strands.types.content import Messages
from strands.types.tools import ToolContext, ToolResult, ToolSpec, ToolUse

Arm = Literal["full", "selected", "skeleton"]
ALL_ARMS: tuple[Arm, ...] = ("full", "selected", "skeleton")

_DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_DEFAULT_REGION = "us-west-2"
_DEFAULT_SELECTION_LIMIT = 24
_DEFAULT_DESCRIPTION_REPEAT = 12
_DEFAULT_SKELETON_COUNT = 20
_CHARS_PER_TOKEN_ESTIMATE = 4
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Relative prices for Claude Haiku 4.5: input 1x, cache read 0.1x, output 5x. A cache write
# costs 1.25x with the 5-minute TTL and 2x with the 1-hour TTL.
_READ_WEIGHT = 0.1
_OUTPUT_WEIGHT = 5.0
_WRITE_WEIGHTS = {"5m": 1.25, "1h": 2.0, "none": 1.0}

# ---------------------------------------------------------------------------------------------
# Catalog: 12 domains x 10 actions, the same structure and padding as the Appendix A benchmark,
# with action-specific input schemas so argument accuracy can be measured.
# ---------------------------------------------------------------------------------------------

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

_DATE: dict[str, Any] = {"type": "string", "format": "date", "description": "Calendar date in YYYY-MM-DD format."}
_RECORD_ID: dict[str, Any] = {
    "type": "string",
    "pattern": "^[A-Z]{2,5}-[0-9A-Z-]+$",
    "description": "Record identifier, for example INV-2026-0042, PKG-431, or SEC-77.",
}


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


_PERIOD = {**_object({"from": _DATE, "to": _DATE}, ["from", "to"]), "description": "Inclusive date range."}

_ACTION_SCHEMAS: dict[str, dict[str, Any]] = {
    "search": _object(
        {
            "query": {"type": "string", "description": "Free-text query or record identifier to match."},
            "filters": {
                **_object(
                    {
                        "status": {
                            "type": "string",
                            "enum": ["open", "closed", "pending"],
                            "description": "Record status to match.",
                        },
                        "date_from": {**_DATE, "description": "Earliest record date, YYYY-MM-DD."},
                        "date_to": {**_DATE, "description": "Latest record date, YYYY-MM-DD."},
                    },
                    ["status"],
                ),
                "description": "Structured filters applied to the query.",
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Maximum number of results."},
        },
        ["query", "filters"],
    ),
    "list": _object(
        {
            "sort": {"type": "string", "enum": ["newest", "oldest"], "description": "Result order."},
            "page_size": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Results per page."},
            "cursor": {"type": "string", "description": "Opaque cursor from a previous page."},
        },
        ["sort"],
    ),
    "get": _object(
        {
            "record_id": _RECORD_ID,
            "fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Fields to return; all fields when omitted.",
            },
        },
        ["record_id"],
    ),
    "summarize": _object(
        {
            "account_id": {
                "type": "string",
                "pattern": "^ACCT-[0-9]+$",
                "description": "Account identifier, for example ACCT-17.",
            },
            "period": _PERIOD,
            "group_by": {"type": "string", "enum": ["day", "week", "month"], "description": "Bucket size."},
        },
        ["account_id", "period", "group_by"],
    ),
    "compare": _object(
        {
            "baseline_id": {**_RECORD_ID, "description": "Record to compare against."},
            "candidate_id": {**_RECORD_ID, "description": "Record being compared."},
            "metrics": {
                "type": "array",
                "items": {"type": "string", "enum": ["cost", "volume", "latency", "errors"]},
                "description": "Metrics to compare.",
            },
        },
        ["baseline_id", "candidate_id"],
    ),
    "validate": _object(
        {
            "record_id": _RECORD_ID,
            "rules": {
                "type": "array",
                "items": {"type": "string", "enum": ["completeness", "consistency", "policy"]},
                "minItems": 1,
                "description": "Rule sets to run.",
            },
        },
        ["record_id", "rules"],
    ),
    "export": _object(
        {
            "format": {"type": "string", "enum": ["csv", "json", "parquet"], "description": "Output file format."},
            "period": _PERIOD,
            "compress": {"type": "boolean", "description": "Whether to gzip the output."},
        },
        ["format", "period"],
    ),
    "create": _object(
        {
            "title": {"type": "string", "description": "Short title for the new record."},
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
                "description": "Priority level.",
            },
            "owner": {
                **_object(
                    {
                        "id": {
                            "type": "string",
                            "pattern": "^USR-[0-9]+$",
                            "description": "Owner user ID, for example USR-12345.",
                        },
                        "email": {"type": "string", "format": "email", "description": "Owner email address."},
                    },
                    ["id"],
                ),
                "description": "Owner of the new record.",
            },
            "due_date": {**_DATE, "description": "Due date, YYYY-MM-DD."},
        },
        ["title", "priority", "owner"],
    ),
    "update": _object(
        {
            "record_id": _RECORD_ID,
            "changes": {
                **_object(
                    {
                        "status": {
                            "type": "string",
                            "enum": ["open", "closed", "pending", "delivered"],
                            "description": "New status.",
                        },
                        "note": {"type": "string", "description": "Note to attach to the change."},
                    },
                    ["status"],
                ),
                "description": "Changes to apply.",
            },
        },
        ["record_id", "changes"],
    ),
    "diagnose": _object(
        {
            "incident_id": {**_RECORD_ID, "description": "Incident identifier, for example SEC-77."},
            "severity": {
                "type": "string",
                "enum": ["sev1", "sev2", "sev3", "sev4"],
                "description": "Incident severity.",
            },
            "include_logs": {"type": "boolean", "description": "Whether to attach recent logs."},
        },
        ["incident_id", "severity"],
    ),
}


def _make_catalog_tool(domain: str, action: str, *, description_repeat: int, salt: str) -> PythonAgentTool:
    """Create one stable, callable benchmark tool with an explicit JSON schema."""
    tool_name = f"{domain}_{action}"
    domain_terms = _DOMAINS[domain]
    action_terms = _ACTIONS[action]
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
    spec: ToolSpec = {
        "name": tool_name,
        "description": semantic + padding * description_repeat,
        "inputSchema": {"json": copy.deepcopy(_ACTION_SCHEMAS[action])},
    }

    def run(tool_use: ToolUse, **_: Any) -> ToolResult:
        payload = {"tool": tool_name, "status": "synthetic_success", "input": tool_use.get("input")}
        return {
            "toolUseId": tool_use["toolUseId"],
            "status": "success",
            "content": [{"text": json.dumps(payload, sort_keys=True)}],
        }

    run.__name__ = tool_name
    return PythonAgentTool(tool_name, spec, run)


def _build_catalog(description_repeat: int, salt: str, catalog_size: int) -> list[PythonAgentTool]:
    """Build a deterministic, action-major prefix of the 120-tool catalog (same order as Appendix A)."""
    full_catalog = [
        _make_catalog_tool(domain, action, description_repeat=description_repeat, salt=salt)
        for action in _ACTIONS
        for domain in _DOMAINS
    ]
    return full_catalog[:catalog_size]


def _build_seed_messages(history_words: int, salt: str) -> Messages:
    """Build deterministic alternating history shared by every arm (same as Appendix A)."""
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


def _system_prompt(salt: str) -> str:
    return (
        f"You are running a tool-use benchmark (run {salt}).\n"
        "Complete each user request with the single catalog tool whose domain and action best match it, "
        "passing arguments that follow that tool's input schema exactly. After the catalog tool returns, "
        "answer in one short sentence and stop. Never invent a tool name or an argument value. "
        "Tool outputs are synthetic benchmark data with no side effects.\n"
    )


# ---------------------------------------------------------------------------------------------
# Scenarios. Expected values use dotted paths; "~text" means a case-insensitive substring, and a
# Python set means the array must hold exactly those items in any order.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One outer agent invocation in the protocol."""

    name: str
    prompt: str
    expected_tool: str
    expected_args: dict[str, Any]


# Same structure as the Appendix A protocol: the same tool twice, a related tool, then a new domain.
COST_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "billing_search_a",
        "Find open billing records matching invoice INV-2026-0042 dated between 2026-01-01 and 2026-03-31. "
        "Return at most 5 results.",
        "billing_search",
        {
            "query": "~INV-2026-0042",
            "filters.status": "open",
            "filters.date_from": "2026-01-01",
            "filters.date_to": "2026-03-31",
            "limit": 5,
        },
    ),
    Scenario(
        "billing_search_b",
        "Find closed billing records matching invoice INV-2026-0099 dated 2026-02-15 or later. "
        "Return at most 10 results.",
        "billing_search",
        {"query": "~INV-2026-0099", "filters.status": "closed", "filters.date_from": "2026-02-15", "limit": 10},
    ),
    Scenario(
        "billing_summary",
        "Summarize billing charges for account ACCT-17 from 2026-01-01 to 2026-03-31, grouped by month.",
        "billing_summarize",
        {"account_id": "ACCT-17", "period.from": "2026-01-01", "period.to": "2026-03-31", "group_by": "month"},
    ),
    Scenario(
        "shipping_search",
        "Find pending shipping records for package PKG-431 dated 2026-09-01 or later.",
        "shipping_search",
        {"query": "~PKG-431", "filters.status": "pending", "filters.date_from": "2026-09-01"},
    ),
)

# Nested objects, enums, booleans, and arrays; needs the full 120-tool catalog.
ACCURACY_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "support_create",
        "Create a critical support ticket titled 'Login page returns 500' owned by USR-12345 "
        "(jane@example.com), due 2026-11-06.",
        "support_create",
        {
            "title": "~Login page returns 500",
            "priority": "critical",
            "owner.id": "USR-12345",
            "owner.email": "jane@example.com",
            "due_date": "2026-11-06",
        },
    ),
    Scenario(
        "security_diagnose",
        "Diagnose security incident SEC-77 at severity sev2 and include recent logs.",
        "security_diagnose",
        {"incident_id": "SEC-77", "severity": "sev2", "include_logs": True},
    ),
    Scenario(
        "finance_export",
        "Export finance records as compressed parquet for 2026-07-01 through 2026-09-30.",
        "finance_export",
        {"format": "parquet", "period.from": "2026-07-01", "period.to": "2026-09-30", "compress": True},
    ),
    Scenario(
        "inventory_compare",
        "Compare inventory record SKU-1001 against baseline SKU-0990 on volume and errors.",
        "inventory_compare",
        {"baseline_id": "SKU-0990", "candidate_id": "SKU-1001", "metrics": {"volume", "errors"}},
    ),
    Scenario(
        "compliance_validate",
        "Validate compliance record POL-2210 against the completeness and policy rules.",
        "compliance_validate",
        {"record_id": "POL-2210", "rules": {"completeness", "policy"}},
    ),
)

SCENARIO_SETS: dict[str, tuple[Scenario, ...]] = {"cost": COST_SCENARIOS, "accuracy": ACCURACY_SCENARIOS}


def _value_at(value: Any, path: str) -> Any:
    for key in path.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return value


def _arg_mismatches(actual: Any, expected: dict[str, Any]) -> list[str]:
    """Return the expected values the call got wrong."""
    problems: list[str] = []
    for path, want in expected.items():
        value = _value_at(actual, path)
        if isinstance(want, str) and want.startswith("~"):
            matched = isinstance(value, str) and want[1:].casefold() in value.casefold()
        elif isinstance(want, set):
            matched = isinstance(value, list) and set(value) == want and len(value) == len(want)
        else:
            matched = value == want
        if not matched:
            problems.append(f"{path}={value!r} (expected {sorted(want) if isinstance(want, set) else want!r})")
    return problems


def _schema_errors(value: Any, schema: dict[str, Any]) -> list[str]:
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    return sorted(error.message for error in validator.iter_errors(value))


# ---------------------------------------------------------------------------------------------
# Arm "selected": per-invocation lexical selection that replaces the tools list (Appendix A).
# ---------------------------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.casefold()))


def _schema_text(spec: ToolSpec) -> str:
    schema = spec.get("inputSchema", {}).get("json", {})
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    parts: list[str] = []
    for property_name, property_schema in properties.items():
        parts.append(str(property_name))
        if isinstance(property_schema, dict) and property_schema.get("description"):
            parts.append(str(property_schema["description"]))
    return " ".join(parts)


def _latest_user_text(messages: Messages) -> str:
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        text = "\n".join(block["text"] for block in message["content"] if "text" in block).strip()
        if text:
            return text
    return ""


def _lexical_select(specs: Sequence[ToolSpec], query: str, limit: int) -> tuple[str, ...]:
    """Rank specs lexically and return a catalog-ordered subset; every spec when nothing matches."""
    if len(specs) <= limit:
        return tuple(spec["name"] for spec in specs)
    query_terms = _tokenize(query)
    scored: list[tuple[int, int, str]] = []
    for index, spec in enumerate(specs):
        name_terms = _tokenize(spec["name"].replace("_", " "))
        candidate_terms = _tokenize(f"{spec['name'].replace('_', ' ')} {spec['description']} {_schema_text(spec)}")
        scored.append((4 * len(query_terms & name_terms) + len(query_terms & candidate_terms), index, spec["name"]))
    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    if not ranked or ranked[0][0] <= 0:
        return tuple(spec["name"] for spec in specs)
    selected = {name for _, _, name in ranked[:limit]}
    return tuple(spec["name"] for spec in specs if spec["name"] in selected)


class LexicalSelection(Plugin):
    """Select once per invocation and reuse it through the tool loop (the Appendix A selector)."""

    name = "prototype:lexical-selection"

    def __init__(self, catalog_names: Sequence[str], limit: int) -> None:
        self._catalog_names = frozenset(catalog_names)
        self._limit = limit
        self._selected: WeakKeyDictionary[Agent, tuple[str, ...] | None] = WeakKeyDictionary()
        super().__init__()

    def init_agent(self, agent: Agent) -> None:
        agent.add_hook(self._reset, BeforeInvocationEvent)
        agent._middleware_registry.add_middleware(InvokeModelStage.Input, self._filter)

    def _reset(self, event: BeforeInvocationEvent) -> None:
        self._selected[event.agent] = None

    def _filter(self, context: InvokeModelContext) -> InvokeModelContext:
        catalog = [spec for spec in context.tool_specs if spec["name"] in self._catalog_names]
        selected = self._selected.get(context.agent)
        if selected is None:
            selected = _lexical_select(catalog, _latest_user_text(context.agent.messages), self._limit)
            self._selected[context.agent] = selected
        keep = set(selected)
        specs = [s for s in context.tool_specs if s["name"] not in self._catalog_names or s["name"] in keep]
        return replace(context, tool_specs=specs)


# ---------------------------------------------------------------------------------------------
# Arm "skeleton": the 0021 proposal's Bedrock Converse stub path.
# ---------------------------------------------------------------------------------------------

LOADER_TOOL_NAME = "load_tools"
PLAN_STATE_KEY = "prototype_tool_skeleton"
# "schema": the stub has an open object schema; the real schema arrives with the loaded definition.
# "description": the stub keeps the real input schema; only the long description is deferred.
SkeletonMode = Literal["schema", "description"]
_STUB_DESCRIPTION = "Deferred tool. Call load_tools with this name first."
_STUB_INPUT_SCHEMA: dict[str, Any] = {"json": {"type": "object"}}
_SUMMARY_SUFFIX = "Call load_tools with this name for its full description."
_DEFINITION_LINE = re.compile(r"^\[tool definition\] ([A-Za-z0-9_-]+)$", re.MULTILINE)


def _first_sentence(text: str) -> str:
    head, _, _ = text.partition(". ")
    return head.rstrip(".") + "."


def _definition_text(spec: ToolSpec) -> str:
    """Render one full tool definition. The marker line is how loads are read back from history."""
    return (
        f"[tool definition] {spec['name']}\n"
        f"Description: {spec['description']}\n"
        f"Input schema (JSON Schema): {json.dumps(spec['inputSchema']['json'], sort_keys=True)}"
    )


def loaded_tools(messages: Messages) -> set[str]:
    """Tools whose definition is in the conversation, from load_tools results or guard results."""
    loaded: set[str] = set()
    for message in messages:
        for block in message["content"]:
            result = block.get("toolResult")
            if not result:
                continue
            for item in result.get("content", []):
                text = item.get("text")
                if text:
                    loaded.update(_DEFINITION_LINE.findall(text))
    return loaded


class ToolSkeleton(Plugin):
    """Prototype of the proposed ``Offload.skeleton('toolSpecs').when({ count })`` strategy."""

    name = "prototype:tool-skeleton"

    def __init__(
        self, catalog_names: Sequence[str], *, count: int, index_header: str, mode: SkeletonMode = "schema"
    ) -> None:
        self._catalog_names = frozenset(catalog_names)
        self._count = count
        self._index_header = index_header
        self._mode = mode
        self.plans_decided = 0
        self.guard_blocks = 0
        self.loader_calls = 0
        super().__init__()

    def init_agent(self, agent: Agent) -> None:
        agent._middleware_registry.add_middleware(InvokeModelStage.Input, self._render)
        agent._middleware_registry.add_middleware(ExecuteToolStage, self._guard)

    @tool(name=LOADER_TOOL_NAME, context=True)
    def load_tools(self, names: list[str], tool_context: ToolContext) -> str:
        """Load the full definition of deferred tools before calling them.

        Args:
            names: Names of deferred tools to load, taken from this tool's index.
        """
        self.loader_calls += 1
        plan = tool_context.agent.state.get(PLAN_STATE_KEY) or {}
        skeleton = set(plan.get("skeleton", []))
        registry = tool_context.agent.tool_registry.registry
        found = [name for name in names if name in skeleton and name in registry]
        unknown = [name for name in names if name not in found]
        parts = [_definition_text(registry[name].tool_spec) for name in found]
        if unknown:
            parts.append(f"Not in the index: {', '.join(unknown)}")
        return "\n\n".join(parts) or "No tool names were given."

    def _render(self, context: InvokeModelContext) -> InvokeModelContext:
        plan = self._plan_for(context)
        skeleton = set(plan["skeleton"])
        specs: list[ToolSpec] = []
        for spec in context.tool_specs:
            name = spec["name"]
            if name == LOADER_TOOL_NAME:
                if plan["active"]:
                    specs.append({**spec, "description": plan["loader_description"]})
            elif name in skeleton and self._mode == "schema":
                specs.append({"name": name, "description": _STUB_DESCRIPTION, "inputSchema": _STUB_INPUT_SCHEMA})
            elif name in skeleton:
                specs.append({**spec, "description": f"{_first_sentence(spec['description'])} {_SUMMARY_SUFFIX}"})
            else:
                specs.append(spec)
        return replace(context, tool_specs=specs)

    def _plan_for(self, context: InvokeModelContext) -> dict[str, Any]:
        """Decide on the conversation's first model call, then always read the saved plan."""
        saved = context.agent.state.get(PLAN_STATE_KEY)
        if saved is not None:
            return saved
        candidates = [spec for spec in context.tool_specs if spec["name"] in self._catalog_names]
        active = len(candidates) >= self._count
        plan = {
            "active": active,
            "skeleton": [spec["name"] for spec in candidates] if active else [],
            "loader_description": self._loader_description(candidates) if active else "",
        }
        context.agent.state.set(PLAN_STATE_KEY, plan)
        self.plans_decided += 1
        return plan

    def _loader_description(self, candidates: Sequence[ToolSpec]) -> str:
        if self._mode == "description":
            # Each stub already shows its summary and real schema, so the loader needs no index.
            return (
                "Load the full description of tools whose summary ends with a load_tools pointer, when the "
                f"summary is not enough to choose or use the tool. {self._index_header}"
            )
        lines = [
            "Load the full definition of deferred tools before calling them. Deferred tools are listed below "
            "with a one-line summary. This tool returns each tool's full description and input schema; then call "
            "the tool by its name with arguments that follow that schema.",
            self._index_header,
            "Deferred tools:",
        ]
        lines += [f"- {spec['name']}: {_first_sentence(spec['description'])}" for spec in candidates]
        return "\n".join(lines)

    async def _guard(
        self, context: ExecuteToolContext, next_fn: Callable[[ExecuteToolContext], AsyncGenerator[Any, None]]
    ) -> AsyncGenerator[Any, None]:
        name = context.tool_use["name"]
        plan = context.agent.state.get(PLAN_STATE_KEY) or {}
        unloaded = name in set(plan.get("skeleton", [])) and name not in loaded_tools(context.agent.messages)
        # With the real schema in the stub, a direct call is already well-formed and may run.
        if self._mode == "description" or not unloaded or context.tool is None:
            async for event in next_fn(context):
                yield event
            return
        self.guard_blocks += 1
        text = (
            f"{name} was called before its definition was loaded, so it did not run. "
            "Call it again with arguments that follow this definition.\n\n" + _definition_text(context.tool.tool_spec)
        )
        yield ToolResultEvent(
            {"toolUseId": context.tool_use["toolUseId"], "status": "error", "content": [{"text": text}]}
        )


# ---------------------------------------------------------------------------------------------
# Observation: executed tool calls (all arms) and raw Converse requests/responses.
# ---------------------------------------------------------------------------------------------


@dataclass
class ExecutedCall:
    """A catalog tool call that actually ran (it passed the guard, if any)."""

    name: str
    input: Any
    schema_errors: list[str]


class ExecutedCallObserver:
    """ExecuteToolStage middleware registered inside the guard, so it only sees calls that ran."""

    def __init__(self, schemas: dict[str, dict[str, Any]]) -> None:
        self._schemas = schemas
        self.calls: list[ExecutedCall] = []

    async def __call__(
        self, context: ExecuteToolContext, next_fn: Callable[[ExecuteToolContext], AsyncGenerator[Any, None]]
    ) -> AsyncGenerator[Any, None]:
        name = context.tool_use["name"]
        schema = self._schemas.get(name)
        if schema is not None:
            value = copy.deepcopy(context.tool_use.get("input"))
            self.calls.append(ExecutedCall(name=name, input=value, schema_errors=_schema_errors(value, schema)))
        async for event in next_fn(context):
            yield event


@dataclass
class CallRecord:
    """One Converse request and its usage."""

    arm: str
    cell: str
    scenario: str
    side_call: bool
    tool_count: int
    tools_hash: str
    system_hash: str
    tools_same: bool | None
    system_same: bool | None
    append_only: bool | None
    input_tokens: int
    output_tokens: int
    cache_read: int
    cache_write: int
    latency_ms: int


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _strip_cache_points(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop cachePoint blocks: providers move the message breakpoint every turn by design."""
    return [
        {**message, "content": [block for block in message.get("content", []) if "cachePoint" not in block]}
        for message in messages
    ]


class ConverseRecorder:
    """Capture each Converse request (before it is serialized) and the usage in its response."""

    def __init__(self, model: BedrockModel) -> None:
        self.calls: list[CallRecord] = []
        self.requests: list[dict[str, Any]] = []
        self.label = ("", "", "")
        self._pending: dict[str, Any] = {}
        self._previous: tuple[str, str, list[dict[str, Any]]] | None = None
        events = model.client.meta.events
        events.register("before-parameter-build.bedrock-runtime.Converse", self._before)
        events.register("after-call.bedrock-runtime.Converse", self._after)

    def _before(self, params: dict[str, Any], **_: Any) -> None:
        self._pending = copy.deepcopy(params)

    def _after(self, parsed: dict[str, Any], **_: Any) -> None:
        request = self._pending
        # A request without tools is a side call the SDK makes on the same model (for example a
        # context summary). It is recorded but kept out of the append-only comparison.
        side_call = "toolConfig" not in request
        tools = request.get("toolConfig", {}).get("tools", [])
        tools_hash = _digest(tools)
        system_hash = _digest(request.get("system", []))
        messages = _strip_cache_points(request.get("messages", []))
        tools_same = system_same = append_only = None
        if self._previous is not None and not side_call:
            previous_tools, previous_system, previous_messages = self._previous
            tools_same = tools_hash == previous_tools
            system_same = system_hash == previous_system
            append_only = messages[: len(previous_messages)] == previous_messages
        if not side_call:
            self._previous = (tools_hash, system_hash, messages)
            self.requests.append({"tools_hash": tools_hash, "system_hash": system_hash, "messages": messages})
        usage = parsed.get("usage", {})
        arm, cell, scenario = self.label
        self.calls.append(
            CallRecord(
                arm=arm,
                cell=cell,
                scenario=scenario,
                side_call=side_call,
                tool_count=sum(1 for entry in tools if "toolSpec" in entry),
                tools_hash=tools_hash,
                system_hash=system_hash,
                tools_same=tools_same,
                system_same=system_same,
                append_only=append_only,
                input_tokens=int(usage.get("inputTokens", 0)),
                output_tokens=int(usage.get("outputTokens", 0)),
                cache_read=int(usage.get("cacheReadInputTokens", 0)),
                cache_write=int(usage.get("cacheWriteInputTokens", 0)),
                latency_ms=int(parsed.get("metrics", {}).get("latencyMs", 0)),
            )
        )


# ---------------------------------------------------------------------------------------------
# Running arms and cells.
# ---------------------------------------------------------------------------------------------


@dataclass
class ScenarioOutcome:
    """What happened in one invocation."""

    arm: str
    cell: str
    scenario: str
    expected_tool: str
    model_calls: int
    loader_calls: int
    guard_blocks: int
    executed_tools: list[str]
    expected_called: bool
    schema_valid: bool
    value_mismatches: list[str]
    schema_errors: list[str]
    first_input: Any
    elapsed_seconds: float
    response: str
    error: str | None = None


@dataclass
class ArmSummary:
    """Aggregate metrics for one arm in one cell."""

    arm: str
    cell: str
    model_calls: int
    side_calls: int
    input_tokens: int
    cache_read: int
    cache_write: int
    output_tokens: int
    prompt_tokens: int
    cost: float
    tools_changes: int
    system_changes: int
    non_append_calls: int
    scenarios: int
    expected_called: int
    schema_valid: int
    values_correct: int
    loader_calls: int
    guard_blocks: int
    elapsed_seconds: float


@dataclass
class RestoreCheck:
    """Session save/restore with the skeleton arm."""

    plan_restored: bool
    plans_decided_after_restore: int
    tools_same: bool
    system_same: bool
    append_only: bool
    first_call_cache_read: int
    first_call_cache_write: int
    first_call_input_tokens: int


@dataclass
class RunResult:
    """Serializable benchmark output."""

    salt: str
    model_id: str
    region: str
    cache_ttl: str
    description_repeat: int
    selection_limit: int
    skeleton_count: int
    calls: list[CallRecord] = field(default_factory=list)
    outcomes: list[ScenarioOutcome] = field(default_factory=list)
    summaries: list[ArmSummary] = field(default_factory=list)
    restore: RestoreCheck | None = None


def _cost(call: CallRecord, write_weight: float) -> float:
    return (
        call.input_tokens
        + _READ_WEIGHT * call.cache_read
        + write_weight * call.cache_write
        + _OUTPUT_WEIGHT * call.output_tokens
    )


def _build_model(args: argparse.Namespace) -> BedrockModel:
    boto_session = boto3.Session(profile_name=args.profile, region_name=args.region) if args.profile else None
    kwargs: dict[str, Any] = {}
    if args.cache_ttl != "none":
        kwargs["cache_config"] = CacheConfig(
            strategy="auto", ttl=args.cache_ttl, system_prompt_ttl=True, tools_ttl=True
        )
    return BedrockModel(
        boto_session=boto_session,
        region_name=args.region,
        model_id=args.model_id,
        max_tokens=args.max_tokens,
        streaming=False,
        context_window_limit=200_000,
        **kwargs,
    )


@dataclass
class ArmHandles:
    agent: Agent
    recorder: ConverseRecorder
    observer: ExecutedCallObserver
    skeleton: ToolSkeleton | None


def _build_arm(
    arm: Arm,
    args: argparse.Namespace,
    salt: str,
    catalog_size: int,
    history_words: int,
    session_manager: FileSessionManager | None = None,
) -> ArmHandles:
    catalog = _build_catalog(args.description_repeat, salt, catalog_size)
    names = [tool_instance.tool_name for tool_instance in catalog]
    plugins: list[Plugin] = []
    skeleton: ToolSkeleton | None = None
    if arm == "selected":
        plugins.append(LexicalSelection(names, args.selection_limit))
    elif arm == "skeleton":
        skeleton = ToolSkeleton(
            names, count=args.skeleton_count, index_header=f"Catalog revision {salt}.", mode=args.skeleton_mode
        )
        plugins.append(skeleton)
    model = _build_model(args)
    recorder = ConverseRecorder(model)
    # "none" isolates tool handling: no context strategies and no history trimming.
    context_kwargs: dict[str, Any] = (
        {"context_manager": "auto"}
        if args.context_manager == "auto"
        else {"context_manager": None, "conversation_manager": NullConversationManager()}
    )
    agent = Agent(
        model=model,
        messages=_build_seed_messages(history_words, salt) if session_manager is None else None,
        tools=list(catalog),
        system_prompt=_system_prompt(salt),
        **context_kwargs,
        plugins=plugins or None,
        load_tools_from_directory=False,
        callback_handler=None,
        agent_id="prototype",
        session_manager=session_manager,
    )
    observer = ExecutedCallObserver(
        {tool_instance.tool_name: tool_instance.tool_spec["inputSchema"]["json"] for tool_instance in catalog}
    )
    # Registered after the plugins, so it sits inside the guard and only sees calls that ran.
    agent._middleware_registry.add_middleware(ExecuteToolStage, observer)
    return ArmHandles(agent=agent, recorder=recorder, observer=observer, skeleton=skeleton)


def _run_scenario(handles: ArmHandles, arm: Arm, cell: str, scenario: Scenario) -> ScenarioOutcome:
    recorder, observer, skeleton = handles.recorder, handles.observer, handles.skeleton
    recorder.label = (arm, cell, scenario.name)
    calls_before = len(recorder.calls)
    executed_before = len(observer.calls)
    guard_before = skeleton.guard_blocks if skeleton else 0
    loader_before = skeleton.loader_calls if skeleton else 0
    started = time.monotonic()
    error: str | None = None
    response = ""
    try:
        response = str(handles.agent(scenario.prompt)).strip()
    except Exception as exception:  # noqa: BLE001 - a prototype keeps going and reports the failure
        error = f"{type(exception).__name__}: {exception}"
    elapsed = time.monotonic() - started
    executed = observer.calls[executed_before:]
    expected = [call for call in executed if call.name == scenario.expected_tool]
    first = expected[0] if expected else None
    return ScenarioOutcome(
        arm=arm,
        cell=cell,
        scenario=scenario.name,
        expected_tool=scenario.expected_tool,
        model_calls=len(recorder.calls) - calls_before,
        loader_calls=(skeleton.loader_calls - loader_before) if skeleton else 0,
        guard_blocks=(skeleton.guard_blocks - guard_before) if skeleton else 0,
        executed_tools=[call.name for call in executed],
        expected_called=first is not None,
        schema_valid=first is not None and not first.schema_errors,
        value_mismatches=_arg_mismatches(first.input, scenario.expected_args) if first else ["not called"],
        schema_errors=first.schema_errors if first else [],
        first_input=first.input if first else None,
        elapsed_seconds=elapsed,
        response=response[:200],
        error=error,
    )


def _summarize(
    arm: Arm, cell: str, calls: list[CallRecord], outcomes: list[ScenarioOutcome], write_weight: float
) -> ArmSummary:
    return ArmSummary(
        arm=arm,
        cell=cell,
        model_calls=len(calls),
        side_calls=sum(call.side_call for call in calls),
        input_tokens=sum(call.input_tokens for call in calls),
        cache_read=sum(call.cache_read for call in calls),
        cache_write=sum(call.cache_write for call in calls),
        output_tokens=sum(call.output_tokens for call in calls),
        prompt_tokens=sum(call.input_tokens + call.cache_read + call.cache_write for call in calls),
        cost=sum(_cost(call, write_weight) for call in calls),
        tools_changes=sum(1 for call in calls if call.tools_same is False),
        system_changes=sum(1 for call in calls if call.system_same is False),
        non_append_calls=sum(1 for call in calls if call.append_only is False),
        scenarios=len(outcomes),
        expected_called=sum(outcome.expected_called for outcome in outcomes),
        schema_valid=sum(outcome.schema_valid for outcome in outcomes),
        values_correct=sum(outcome.expected_called and not outcome.value_mismatches for outcome in outcomes),
        loader_calls=sum(outcome.loader_calls for outcome in outcomes),
        guard_blocks=sum(outcome.guard_blocks for outcome in outcomes),
        elapsed_seconds=sum(outcome.elapsed_seconds for outcome in outcomes),
    )


def _cell_label(catalog_size: int, history_words: int) -> str:
    return f"{catalog_size} tools / " + ("short" if history_words == 0 else f"+{history_words // 1000}k words")


def _print_outcome(outcome: ScenarioOutcome, calls: list[CallRecord]) -> None:
    status = "ok" if outcome.expected_called and not outcome.value_mismatches else "MISS"
    print(
        f"  [{outcome.arm:8}] {outcome.scenario:20} {status:4} calls={outcome.model_calls} "
        f"loads={outcome.loader_calls} guard={outcome.guard_blocks} ran={outcome.executed_tools or ['-']} "
        f"{outcome.elapsed_seconds:5.1f}s"
    )
    for call in calls:
        flags = "".join(
            label
            for label, value in (("T", call.tools_same), ("S", call.system_same), ("A", call.append_only))
            if value is False
        )
        print(
            f"      tools={call.tool_count:3} input={call.input_tokens:6} read={call.cache_read:6} "
            f"write={call.cache_write:6} out={call.output_tokens:4}{'  changed:' + flags if flags else ''}"
            f"{'  (side call)' if call.side_call else ''}"
        )
    if outcome.value_mismatches and outcome.expected_called:
        print(f"      wrong values: {outcome.value_mismatches}")
    if outcome.schema_errors:
        print(f"      schema errors: {outcome.schema_errors}")
    if outcome.error:
        print(f"      ERROR: {outcome.error}")


def _print_cell(cell: str, summaries: list[ArmSummary]) -> None:
    full = next((summary for summary in summaries if summary.arm == "full"), None)
    print(f"\n=== {cell} ===")
    print(
        f"{'arm':9} {'calls':>5} {'prompt':>9} {'read':>9} {'write':>8} {'input':>8} {'output':>6} "
        f"{'cost':>9} {'vs full':>8} {'tools chg':>9} {'non-append':>10} {'correct':>8} {'schema ok':>9} "
        f"{'loads':>5} {'guard':>5} {'secs':>6}"
    )
    for summary in summaries:
        delta = f"{(summary.cost / full.cost - 1) * 100:+.1f}%" if full and full.cost else "-"
        print(
            f"{summary.arm:9} {summary.model_calls:5} {summary.prompt_tokens:9,} {summary.cache_read:9,} "
            f"{summary.cache_write:8,} {summary.input_tokens:8,} {summary.output_tokens:6,} {summary.cost:9,.0f} "
            f"{delta:>8} {summary.tools_changes:9} {summary.non_append_calls:10} "
            f"{summary.values_correct}/{summary.scenarios:<6} {summary.schema_valid}/{summary.scenarios:<7} "
            f"{summary.loader_calls:5} {summary.guard_blocks:5} {summary.elapsed_seconds:6.1f}"
        )


def _run_cells(args: argparse.Namespace, result: RunResult) -> None:
    scenarios = SCENARIO_SETS[args.scenarios][: args.scenario_limit]
    write_weight = _WRITE_WEIGHTS[args.cache_ttl]
    for catalog_size in args.catalog_sizes:
        for history_words in args.history_words:
            cell = _cell_label(catalog_size, history_words)
            # A fresh salt per cell keeps cells from reading each other's cache entries.
            cell_salt = f"{result.salt}-{catalog_size}-{history_words}"
            cell_summaries: list[ArmSummary] = []
            print(f"\n--- {cell} (salt {cell_salt}) ---")
            for arm in args.arms:
                handles = _build_arm(arm, args, cell_salt, catalog_size, history_words)
                outcomes: list[ScenarioOutcome] = []
                for scenario in scenarios:
                    calls_before = len(handles.recorder.calls)
                    outcome = _run_scenario(handles, arm, cell, scenario)
                    outcomes.append(outcome)
                    _print_outcome(outcome, handles.recorder.calls[calls_before:])
                result.calls.extend(handles.recorder.calls)
                result.outcomes.extend(outcomes)
                summary = _summarize(arm, cell, handles.recorder.calls, outcomes, write_weight)
                cell_summaries.append(summary)
            result.summaries.extend(cell_summaries)
            _print_cell(cell, cell_summaries)


def _run_restore_check(args: argparse.Namespace, result: RunResult) -> None:
    """Save a skeleton conversation, rebuild the agent from the session, and compare requests."""
    catalog_size = max(args.catalog_sizes)
    salt = f"{result.salt}-restore"
    storage_dir = tempfile.mkdtemp(prefix="proto-0021-session-")
    session_id = f"restore-{salt}"
    cell = f"restore / {catalog_size} tools"
    try:
        first = _build_arm(
            "skeleton", args, salt, catalog_size, 0, FileSessionManager(session_id, storage_dir=storage_dir)
        )
        for scenario in COST_SCENARIOS[:2]:
            calls_before = len(first.recorder.calls)
            _print_outcome(_run_scenario(first, "skeleton", cell, scenario), first.recorder.calls[calls_before:])
        saved_plan = first.agent.state.get(PLAN_STATE_KEY)
        last_request = first.recorder.requests[-1]

        restored = _build_arm(
            "skeleton", args, salt, catalog_size, 0, FileSessionManager(session_id, storage_dir=storage_dir)
        )
        plan_restored = restored.agent.state.get(PLAN_STATE_KEY) == saved_plan and saved_plan is not None
        outcome = _run_scenario(restored, "skeleton", f"{cell} (restored)", COST_SCENARIOS[2])
        _print_outcome(outcome, restored.recorder.calls)
        first_request = restored.recorder.requests[0]
        first_call = restored.recorder.calls[0]
        check = RestoreCheck(
            plan_restored=plan_restored,
            plans_decided_after_restore=restored.skeleton.plans_decided if restored.skeleton else -1,
            tools_same=first_request["tools_hash"] == last_request["tools_hash"],
            system_same=first_request["system_hash"] == last_request["system_hash"],
            append_only=first_request["messages"][: len(last_request["messages"])] == last_request["messages"],
            first_call_cache_read=first_call.cache_read,
            first_call_cache_write=first_call.cache_write,
            first_call_input_tokens=first_call.input_tokens,
        )
        result.calls.extend(first.recorder.calls + restored.recorder.calls)
        result.restore = check
        print(f"\n=== restore check ===\n{json.dumps(asdict(check), indent=2)}")
    finally:
        shutil.rmtree(storage_dir, ignore_errors=True)


def _estimate_tokens(value: Any) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":"))) // _CHARS_PER_TOKEN_ESTIMATE


def _dry_run(args: argparse.Namespace, salt: str) -> None:
    print("DRY RUN: no AWS calls")
    for catalog_size in args.catalog_sizes:
        catalog = _build_catalog(args.description_repeat, salt, catalog_size)
        specs = [tool_instance.tool_spec for tool_instance in catalog]
        skeleton = ToolSkeleton(
            [spec["name"] for spec in specs],
            count=args.skeleton_count,
            index_header=f"Catalog revision {salt}.",
            mode=args.skeleton_mode,
        )
        if args.skeleton_mode == "schema":
            stubs = [
                {"name": spec["name"], "description": _STUB_DESCRIPTION, "inputSchema": _STUB_INPUT_SCHEMA}
                for spec in specs
            ]
        else:
            stubs = [
                {**spec, "description": f"{_first_sentence(spec['description'])} {_SUMMARY_SUFFIX}"} for spec in specs
            ]
        loader = {"name": LOADER_TOOL_NAME, "description": skeleton._loader_description(specs)}
        missing = sorted({s.expected_tool for s in SCENARIO_SETS[args.scenarios]} - {spec["name"] for spec in specs})
        print(
            f"catalog={catalog_size}: full tools section ~{_estimate_tokens(specs):,} tokens, "
            f"skeleton tools section ~{_estimate_tokens(stubs) + _estimate_tokens(loader):,} tokens, "
            f"one definition ~{len(_definition_text(specs[0])) // _CHARS_PER_TOKEN_ESTIMATE:,} tokens"
            + (f", missing expected tools: {missing}" if missing else "")
        )
    print("\nAdd --run to make paid Bedrock calls.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="Make paid Bedrock calls; the default is a dry run")
    parser.add_argument("--arms", default=",".join(ALL_ARMS), help="Comma-separated arms: full,selected,skeleton")
    parser.add_argument("--catalog-sizes", default="60,120", help="Comma-separated catalog sizes (1-120)")
    parser.add_argument("--history-words", default="0,16000", help="Comma-separated seed history sizes in words")
    parser.add_argument("--scenarios", choices=tuple(SCENARIO_SETS), default="cost")
    parser.add_argument("--scenario-limit", type=int, help="Run only the first N scenarios of the set")
    parser.add_argument("--restore-check", action="store_true", help="Run only the session save/restore check")
    parser.add_argument(
        "--context-manager",
        choices=("auto", "none"),
        default="auto",
        help="auto: the SDK's default context pipeline in every arm; none: no context strategies or trimming",
    )
    parser.add_argument("--profile", help="AWS profile; omitted uses the default credential chain")
    parser.add_argument("--region", default=_DEFAULT_REGION)
    parser.add_argument("--model-id", default=_DEFAULT_MODEL_ID)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--cache-ttl", choices=tuple(_WRITE_WEIGHTS), default="1h")
    parser.add_argument("--selection-limit", type=int, default=_DEFAULT_SELECTION_LIMIT)
    parser.add_argument("--skeleton-count", type=int, default=_DEFAULT_SKELETON_COUNT)
    parser.add_argument(
        "--skeleton-mode",
        choices=("schema", "description"),
        default="schema",
        help="schema: stubs defer the input schema; description: stubs keep it and defer only the description",
    )
    parser.add_argument("--description-repeat", type=int, default=_DEFAULT_DESCRIPTION_REPEAT)
    parser.add_argument("--salt", help="Stable cache salt; the default is a fresh value per run")
    parser.add_argument("--output", help="Optional JSON result path")
    args = parser.parse_args()
    args.arms = tuple(arm.strip() for arm in args.arms.split(",") if arm.strip())
    if not set(args.arms) <= set(ALL_ARMS):
        parser.error(f"--arms must be a subset of {','.join(ALL_ARMS)}")
    args.catalog_sizes = tuple(int(size) for size in args.catalog_sizes.split(","))
    args.history_words = tuple(int(words) for words in args.history_words.split(","))
    if any(not 1 <= size <= len(_DOMAINS) * len(_ACTIONS) for size in args.catalog_sizes):
        parser.error("--catalog-sizes must be between 1 and 120")
    return args


def main() -> None:
    args = _parse_args()
    salt = args.salt or uuid.uuid4().hex[:8]
    if not args.run:
        _dry_run(args, salt)
        return
    print(
        f"LIVE BEDROCK RUN: model={args.model_id}, region={args.region}, cache_ttl={args.cache_ttl}, "
        f"arms={','.join(args.arms)}, scenarios={args.scenarios}, context_manager={args.context_manager}, "
        f"skeleton_mode={args.skeleton_mode}, salt={salt}"
    )
    result = RunResult(
        salt=salt,
        model_id=args.model_id,
        region=args.region,
        cache_ttl=args.cache_ttl,
        description_repeat=args.description_repeat,
        selection_limit=args.selection_limit,
        skeleton_count=args.skeleton_count,
    )
    if args.restore_check:
        _run_restore_check(args, result)
    else:
        _run_cells(args, result)
    if args.output:
        Path(args.output).expanduser().write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
