"""Built-in middleware stages and their context/result types."""

from __future__ import annotations

import dataclasses
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from ..interrupt import _AGENT_STREAM_INTERRUPT_ID_PREFIX, Interrupt, InterruptException
from .types import MiddlewareStage

if TYPE_CHECKING:
    from ..interrupt import _InterruptState
    from ..models.model import Model
    from ..types._events import EventLoopStopEvent, ModelStopReason, ToolResultEvent, TypedEvent
    from ..types.agent import LocalAgent
    from ..types.content import Messages, SystemPrompt
    from ..types.tools import AgentTool, ToolChoice, ToolSpec, ToolUse

_UNSET: Any = object()
"""Default for ``replace()`` keywords, so an explicit ``None`` still counts as a replacement."""

_ContextT = TypeVar("_ContextT", "InvokeModelContext", "ExecuteToolContext")


def _replace(context: _ContextT, **changes: Any) -> _ContextT:
    """Copy ``context`` with every keyword that was actually passed replaced."""
    return dataclasses.replace(context, **{name: value for name, value in changes.items() if value is not _UNSET})


@dataclass
class InvokeModelContext:
    """Context passed to InvokeModelStage middleware.

    The collection fields (messages, system_prompt, tool_specs, tool_choice) are defensive
    copies, so middleware cannot accidentally mutate agent state. ``invocation_state`` and
    ``model`` are instead shared by reference: ``invocation_state`` is the live dict hooks and
    tools write to during streaming, and ``model`` is the model this call invokes (it starts
    as ``agent.model``; middleware may replace it per call).
    """

    agent: LocalAgent
    messages: Messages
    system_prompt: SystemPrompt
    tool_specs: list[ToolSpec]
    tool_choice: ToolChoice | None
    invocation_state: dict[str, Any]
    model: Model
    projected_input_tokens: int | None = None
    dynamic_trailing_blocks: int = 0

    def replace(
        self,
        *,
        messages: Messages = _UNSET,
        system_prompt: SystemPrompt = _UNSET,
        tool_specs: list[ToolSpec] = _UNSET,
        tool_choice: ToolChoice | None = _UNSET,
        invocation_state: dict[str, Any] = _UNSET,
        model: Model = _UNSET,
        projected_input_tokens: int | None = _UNSET,
        dynamic_trailing_blocks: int = _UNSET,
    ) -> InvokeModelContext:
        """Return a copy with the given fields replaced; omitted fields keep their current value.

        Example:
            ```python
            modified = context.replace(system_prompt="Be concise.")
            ```

        Args:
            messages: Messages to send to the model.
            system_prompt: System prompt guiding the model.
            tool_specs: Tool specifications available to the model.
            tool_choice: How the model selects tools.
            invocation_state: Per-invocation state, shared by reference across the run.
            model: The model this call invokes.
            projected_input_tokens: Estimated input token count for this call.
            dynamic_trailing_blocks: Trailing blocks of the last user message rebuilt each call.

        Returns:
            A new ``InvokeModelContext``.
        """
        return _replace(
            self,
            messages=messages,
            system_prompt=system_prompt,
            tool_specs=tool_specs,
            tool_choice=tool_choice,
            invocation_state=invocation_state,
            model=model,
            projected_input_tokens=projected_input_tokens,
            dynamic_trailing_blocks=dynamic_trailing_blocks,
        )


InvokeModelStage: MiddlewareStage[InvokeModelContext, ModelStopReason, TypedEvent] = MiddlewareStage(name="invokeModel")
"""Built-in stage wrapping core model invocation.

Middleware registered for this stage can rate-limit, cache, or transform model inputs/outputs.
"""


@dataclass
class MiddlewareInterruptResult:
    """Value returned by a middleware ``interrupt()`` when the agent resumes.

    Wrapping the response (rather than returning it bare) mirrors the TypeScript SDK and
    leaves room to add fields later without breaking callers.

    Attributes:
        response: The human-provided response the agent resumed with.
    """

    response: Any


def _resolve_middleware_interrupt(
    interrupts: Mapping[str, Interrupt],
    interrupt_id: str,
    name: str,
    reason: Any,
    response: Any,
) -> MiddlewareInterruptResult:
    """Resolve a middleware-initiated interrupt without mutating interrupt state.

    Shared by every ``MiddlewareInterruptible`` context (tool and agent-stream stages). It
    inspects prior responses but never registers the interrupt: the stage's executor
    registers it in its ``InterruptException`` handler as the single source of truth,
    matching the TypeScript SDK where middleware interrupts never write to interrupt state.

    Args:
        interrupts: The prior-response lookup — the agent's live ``interrupts`` dict for the tool
            stage, or a snapshot taken before the pass for the agent-stream stage (whose reads must
            survive a tool cycle ending mid-pass and clearing the live dict).
        interrupt_id: The deterministic id for this interrupt.
        name: User-defined name for the interrupt.
        reason: Optional reason surfaced to the user.
        response: Optional preemptive response — fallback if no prior human response exists.

    Returns:
        The user's response wrapped in a ``MiddlewareInterruptResult``.

    Raises:
        InterruptException: When no response is available yet and none was provided.
    """
    existing = interrupts.get(interrupt_id)
    if existing is not None and existing.response is not None:
        return MiddlewareInterruptResult(response=existing.response)

    if response is not None:
        return MiddlewareInterruptResult(response=response)

    raise InterruptException(Interrupt(id=interrupt_id, name=name, reason=reason))


@dataclass
class ExecuteToolContext:
    """Context passed to ExecuteToolStage middleware.

    ``tool_use`` is a shallow copy of the executor's dict, so reassigning its top-level keys
    cannot corrupt executor state. Its ``input`` value is shared by reference (it can hold
    non-copyable objects, such as the agent injected on direct tool calls), so mutating ``input``
    in place still leaks; build a new ``tool_use`` and pass it through ``replace()`` instead.
    ``invocation_state`` is shared by reference, as hooks receive it.

    ``cancel_signal`` is executor-owned: middleware can observe it, but the tool always receives
    the executor's signal, not the one on this context.

    Supports middleware-initiated interrupts via ``interrupt()`` for human-in-the-loop
    approval flows.
    """

    agent: LocalAgent
    tool: AgentTool | None
    tool_use: ToolUse
    invocation_state: dict[str, Any]
    cancel_signal: threading.Event = field(repr=False)
    _interrupt_state: _InterruptState = field(repr=False)
    """Agent interrupt state that ``interrupt()`` resolves prior responses from; always supplied by the executor."""

    def interrupt(self, name: str, *, reason: Any = None, response: Any = None) -> MiddlewareInterruptResult:
        """Request a human-in-the-loop interrupt.

        On first execution (no prior response) this raises ``InterruptException`` to halt
        the agent. After the user resumes with a response, the second call returns that
        response. Providing ``response`` preemptively skips the interrupt entirely.

        This method is read-only with respect to interrupt state: it inspects prior
        responses but does not register the interrupt itself. The tool executor registers
        it (in its ``InterruptException`` handler) as the single source of truth, matching
        the TypeScript SDK where middleware interrupts never write to interrupt state.

        Args:
            name: User-defined name for the interrupt. The interrupt id is scoped to the tool
                call (``v1:middleware_execute_tool:<toolUseId>:<uuid5(name)>``) but not to the
                individual middleware, so the name must be unique across all middleware that
                interrupt this tool call — two middleware using the same name on the same tool
                call collide and share one response. (This matches the hook/tool interrupt
                contract, which is likewise unique per tool call, not per callback.)
            reason: Optional reason for the interrupt (surfaced to the user).
            response: Optional preemptive response — when set, no interrupt is raised.

        Returns:
            The user's response wrapped in a ``MiddlewareInterruptResult``.

        Raises:
            InterruptException: When no response is available yet and none was provided.
        """
        return _resolve_middleware_interrupt(
            self._interrupt_state.interrupts, self._interrupt_id(name), name, reason, response
        )

    def _interrupt_id(self, name: str) -> str:
        """Derive the interrupt id for ``name``, namespaced by the tool call.

        Follows the SDK's ``v1:`` interrupt-id scheme (see ``types/interrupt.py``), hashing
        the user-provided name so ids stay stable across resumes for the same tool call.
        """
        return f"v1:middleware_execute_tool:{self.tool_use['toolUseId']}:{uuid.uuid5(uuid.NAMESPACE_OID, name)}"

    def replace(
        self,
        *,
        tool: AgentTool | None = _UNSET,
        tool_use: ToolUse = _UNSET,
        invocation_state: dict[str, Any] = _UNSET,
    ) -> ExecuteToolContext:
        """Return a copy with the given fields replaced; omitted fields keep their current value.

        ``agent``, ``cancel_signal``, and the interrupt state are carried over unchanged.

        Example:
            ```python
            modified = context.replace(tool_use={**context.tool_use, "input": cleaned})
            ```

        Args:
            tool: The resolved tool implementation, or ``None`` if not found.
            tool_use: The tool use request (name, toolUseId, input).
            invocation_state: Per-invocation state, shared by reference across the run.

        Returns:
            A new ``ExecuteToolContext``.
        """
        return _replace(self, tool=tool, tool_use=tool_use, invocation_state=invocation_state)


ExecuteToolStage: MiddlewareStage[ExecuteToolContext, ToolResultEvent, TypedEvent] = MiddlewareStage(name="executeTool")
"""Built-in stage wrapping individual tool execution.

Middleware registered for this stage can add telemetry, validate inputs, mock responses,
or gate execution behind a human-in-the-loop interrupt. The result event is the
``ToolResultEvent`` produced by the tool (matching the "last event is the result"
convention used across the SDK).
"""


@dataclass
class AgentStreamContext:
    """Context passed to AgentStreamStage middleware.

    Wraps the entire agent output stream at the outermost interception point, so middleware
    can filter, transform, or inject events across a whole invocation pass, or gate the pass
    behind a human-in-the-loop interrupt.

    ``messages`` is the input for this pass, shared by reference (the executor already
    appended them to history). ``invocation_state`` is likewise shared by reference, matching
    how hooks and tools receive it. The copy-vs-reference contract is not yet finalized;
    middleware that needs isolation should copy explicitly.

    Supports middleware-initiated interrupts via ``interrupt()`` for human-in-the-loop
    approval flows.
    """

    agent: LocalAgent
    messages: Messages
    invocation_state: dict[str, Any]
    # A snapshot of the agent's interrupts taken before the pass, threaded in so interrupt() can
    # resolve prior responses. It is a snapshot rather than the live dict because this context's
    # lifetime spans the whole pass: a tool cycle within the same pass ends and clears the live
    # interrupts, so reading a snapshot keeps a gate's re-read after next_fn stable. Required
    # (the run loop is the sole constructor and always supplies it); excluded from repr to
    # avoid dumping unrelated interrupt bookkeeping.
    _interrupts: Mapping[str, Interrupt] = field(repr=False)

    def interrupt(self, name: str, *, reason: Any = None, response: Any = None) -> MiddlewareInterruptResult:
        """Request a human-in-the-loop interrupt.

        On first execution (no prior response) this raises ``InterruptException`` to halt
        the agent. After the user resumes with a response, the second call returns that
        response. Providing ``response`` preemptively skips the interrupt entirely.

        This method is read-only with respect to interrupt state (see
        ``_resolve_middleware_interrupt``): the run loop registers the interrupt in its
        ``InterruptException`` handler as the single source of truth.

        Args:
            name: User-defined name for the interrupt. The name must be unique
                across all agent-stream middleware that share an interrupt dict — including
                across agents in a Graph/Swarm. Two gates with the same name collide and share
                one response.
            reason: Optional reason for the interrupt (surfaced to the user).
            response: Optional preemptive response — when set, no interrupt is raised.

        Returns:
            The user's response wrapped in a ``MiddlewareInterruptResult``.

        Raises:
            InterruptException: When no response is available yet and none was provided.
            RuntimeError: Raised by the run loop when this is called after the pass has already
                produced its stop event, which resuming cannot replay.
        """
        return _resolve_middleware_interrupt(self._interrupts, self._interrupt_id(name), name, reason, response)

    def _interrupt_id(self, name: str) -> str:
        """Derive the interrupt id for ``name``, namespaced to the agent-stream stage.

        Follows the SDK's ``v1:`` interrupt-id scheme (see ``types/interrupt.py``), hashing
        the user-provided name so ids stay stable across resumes.
        """
        return f"{_AGENT_STREAM_INTERRUPT_ID_PREFIX}{uuid.uuid5(uuid.NAMESPACE_OID, name)}"


AgentStreamStage: MiddlewareStage[AgentStreamContext, EventLoopStopEvent, TypedEvent] = MiddlewareStage(
    name="agentStream"
)
"""Built-in stage wrapping the entire agent output stream (outermost interception point).

Internal: kept out of ``strands.middleware.__all__`` until the copy-vs-reference contract of
``AgentStreamContext`` is finalized. Middleware registered for this stage can filter, transform,
or inject events, short-circuit the whole pass, or gate it behind a human-in-the-loop interrupt.
The result event is the ``EventLoopStopEvent`` that ends the pass.
"""
