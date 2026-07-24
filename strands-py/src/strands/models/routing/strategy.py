"""Routing strategy protocol and the context strategies see when selecting a model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ...types.content import Messages, SystemPrompt
    from ...types.tools import ToolSpec
    from .router import RoutingCandidate


@dataclass(frozen=True)
class RoutingContext:
    """Immutable request data a routing strategy sees when choosing a candidate.

    ``candidates`` are the router's normalized candidates; a strategy inspects them (e.g.
    each concrete model's ``context_window_limit``) and returns one of them.
    """

    messages: Messages
    system_prompt: SystemPrompt
    tool_specs: tuple[ToolSpec, ...]
    candidates: tuple[RoutingCandidate, ...]
    invocation_state: Mapping[str, Any]


@runtime_checkable
class RoutingStrategy(Protocol):
    """How a router chooses a candidate for a call.

    ``select`` runs before the call and returns one of ``context.candidates``. The router
    validates the returned candidate and raises for any other value.
    """

    name: str

    async def select(self, context: RoutingContext) -> RoutingCandidate:
        """Return the candidate to use for this call (one of ``context.candidates``)."""
        ...
