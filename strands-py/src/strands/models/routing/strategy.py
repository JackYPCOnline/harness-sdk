"""Routing strategy protocol and the context passed to it.

A ``RoutingStrategy`` selects the router's candidates in preference order once per invocation, given
the call's messages, system prompt, tool specs, and invocation state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ...types.content import Messages, SystemPrompt
    from ...types.tools import ToolSpec
    from .router import RoutingCandidate


@dataclass(frozen=True)
class RoutingContext:
    """Read-only inputs a strategy sees when selecting candidates.

    The collections are shared by reference and must not be mutated. In a multi-agent run, one
    ``invocation_state`` may be shared across nodes, so its ``"agent"`` value may identify a sibling.
    """

    messages: Messages
    system_prompt: SystemPrompt | None
    tool_specs: Sequence[ToolSpec]
    candidates: Sequence[RoutingCandidate]
    invocation_state: Mapping[str, Any]


@runtime_checkable
class RoutingStrategy(Protocol):
    """Selects candidates for an invocation, most preferred first."""

    async def select(self, context: RoutingContext, **kwargs: Any) -> Sequence[RoutingCandidate]:
        """Return candidates from ``context.candidates`` in preference order.

        Returning a subset is allowed; the router appends the rest in declaration order so fallback
        can still reach every candidate.
        """
        ...
