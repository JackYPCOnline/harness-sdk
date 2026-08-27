"""Middleware system for wrapping agent stages.

Middleware wraps the core stages of an agent run (model invocation and tool execution) with
composable handlers that can transform inputs, transform results, or wrap execution to retry,
cache, short-circuit, or gate behind a human-in-the-loop interrupt. Register handlers with
``agent.add_middleware(stage_or_phase, handler)``.

Each stage exposes three phases with a fixed execution order (Input -> Wrap -> Output ->
terminal), independent of registration order:

- ``Input`` transforms the context before execution.
- ``Output`` transforms the result after execution.
- ``Wrap`` (the bare stage token) wraps the whole operation and calls ``next`` itself.

Example:
    ```python
    from strands import Agent
    from strands.middleware import InvokeModelStage

    agent = Agent()

    async def timing(context, next_fn):
        async for event in next_fn(context):
            yield event

    agent.add_middleware(InvokeModelStage, timing)
    ```
"""

# Per-stage result event types, re-exported so the public surface is self-contained: an
# InvokeModelStage Output handler receives a ModelStopReason, and an ExecuteToolStage handler
# yields a ToolResultEvent to short-circuit or deny a tool call. Their home is types._events
# (a private module), so surface them here rather than making consumers import a private path.
from ..types._events import ModelStopReason, ToolResultEvent
from .stages import (
    ExecuteToolContext,
    ExecuteToolStage,
    InvokeModelContext,
    InvokeModelStage,
    MiddlewareInterruptResult,
)
from .types import (
    MiddlewareHandler,
    MiddlewareInputHandler,
    MiddlewareInputPhase,
    MiddlewareNext,
    MiddlewareOutputHandler,
    MiddlewareOutputPhase,
    MiddlewareResult,
    MiddlewareStage,
    MiddlewareWrapPhase,
)

# AgentStreamStage / AgentStreamContext are intentionally omitted: their copy-vs-reference
# contract is not finalized, so they stay internal (importable from .stages, but not part of
# the public API). This matches the TypeScript SDK, which tags them @internal.
__all__ = [
    "ExecuteToolContext",
    "ExecuteToolStage",
    "InvokeModelContext",
    "InvokeModelStage",
    "MiddlewareHandler",
    "MiddlewareInputHandler",
    "MiddlewareInputPhase",
    "MiddlewareInterruptResult",
    "MiddlewareNext",
    "MiddlewareOutputHandler",
    "MiddlewareOutputPhase",
    "MiddlewareResult",
    "MiddlewareStage",
    "MiddlewareWrapPhase",
    "ModelStopReason",
    "ToolResultEvent",
]
