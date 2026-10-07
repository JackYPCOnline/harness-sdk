"""Middleware type system."""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

TContext = TypeVar("TContext")
TResult = TypeVar("TResult")
TEvent = TypeVar("TEvent")


@runtime_checkable
class InterruptControlEvent(Protocol):
    """Structural type for events that are control-flow signals, never a stage result.

    The middleware registry is stage-agnostic — it must not import tool- or model-specific
    event classes. Any event that declares ``is_interrupt`` (e.g. ``ToolInterruptEvent``)
    matches this protocol, so the Output-phase adapter can recognize an interrupt and keep
    it out of the positional "last event is the result" selection without a coupling import.
    """

    @property
    def is_interrupt(self) -> bool:
        """True when the event halts the stage rather than producing a result."""
        ...


@dataclass
class MiddlewareResult(Generic[TResult]):
    """Wrapper passed to and returned from Output phase handlers.

    The wrapper, rather than the bare result event, gives Output handlers a surface that can
    grow fields without changing the handler signature.

    Attributes:
        value: The stage's result, the last event from the chain (e.g. ``ModelStopReason``).
    """

    value: TResult

    def replace(self, *, value: TResult) -> MiddlewareResult[TResult]:
        """Return a copy with ``value`` replaced.

        Example:
            ```python
            return result.replace(value=transformed_event)
            ```
        """
        return dataclasses.replace(self, value=value)


class MiddlewareInputPhase(Generic[TContext, TResult, TEvent]):
    """Phase sub-token for Input handlers — transforms context before execution."""

    __slots__ = ("_stage", "_phase")

    def __init__(self, stage: MiddlewareStage[TContext, TResult, TEvent]) -> None:
        """Bind this phase sub-token to its parent stage."""
        self._stage = stage
        self._phase = "input"


class MiddlewareWrapPhase(Generic[TContext, TResult, TEvent]):
    """Phase sub-token for Wrap handlers — full async generator wrap."""

    __slots__ = ("_stage", "_phase")

    def __init__(self, stage: MiddlewareStage[TContext, TResult, TEvent]) -> None:
        """Bind this phase sub-token to its parent stage."""
        self._stage = stage
        self._phase = "wrap"


class MiddlewareOutputPhase(Generic[TContext, TResult, TEvent]):
    """Phase sub-token for Output handlers — transforms result after execution."""

    __slots__ = ("_stage", "_phase")

    def __init__(self, stage: MiddlewareStage[TContext, TResult, TEvent]) -> None:
        """Bind this phase sub-token to its parent stage."""
        self._stage = stage
        self._phase = "output"


class MiddlewareStage(Generic[TContext, TResult, TEvent]):
    """A stage token identifying a middleware interception point.

    Only the SDK's built-in tokens (``InvokeModelStage``, ``ExecuteToolStage``) are invoked;
    constructing a custom stage is unsupported.
    """

    __slots__ = ("name", "Input", "Wrap", "Output")

    def __init__(self, name: str) -> None:
        """Create a stage token with its Input/Wrap/Output phase sub-tokens."""
        self.name = name
        self.Input: MiddlewareInputPhase[TContext, TResult, TEvent] = MiddlewareInputPhase(self)
        self.Wrap: MiddlewareWrapPhase[TContext, TResult, TEvent] = MiddlewareWrapPhase(self)
        self.Output: MiddlewareOutputPhase[TContext, TResult, TEvent] = MiddlewareOutputPhase(self)

    def __repr__(self) -> str:
        """Return a debug representation naming the stage."""
        return f"MiddlewareStage(name={self.name!r})"

    def __hash__(self) -> int:
        """Hash by identity so each stage token is a distinct registry key."""
        return id(self)

    def __eq__(self, other: object) -> bool:
        """Compare by identity — a stage token equals only itself."""
        return self is other


# The stage result is the last event a Wrap handler yields, so only Output handlers (which receive
# it explicitly) are generic over TResult.
MiddlewareNext = Callable[[TContext], AsyncGenerator[TEvent, None]]
MiddlewareHandler = Callable[[TContext, MiddlewareNext[TContext, TEvent]], AsyncGenerator[TEvent, None]]
MiddlewareInputHandler = Callable[[TContext], TContext | Awaitable[TContext]]
MiddlewareOutputHandler = Callable[
    [MiddlewareResult[TResult]], MiddlewareResult[TResult] | Awaitable[MiddlewareResult[TResult]]
]
