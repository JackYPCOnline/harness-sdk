"""Routing strategy protocol and the context passed to it.

A ``RoutingStrategy`` runs once per invocation, given the call's messages, system prompt, tool specs,
and invocation state, and returns the candidates to use in preference order. That sequence doubles as
the fallback chain, so the strategy controls whether failover happens at all.
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
        """Return the candidates to use, from ``context.candidates``, most preferred first.

        The returned sequence is also the fallback chain: the router uses the first entry and
        advances through the rest on failure. Omitting a candidate excludes it from fallback, so
        returning a single candidate means the invocation fails rather than switching models.
        """
        ...
