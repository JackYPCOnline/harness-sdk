"""ModelRouter: a reusable, immutable set of candidate models with a routing strategy.

A router is a ``Plugin`` so an agent can accept it through ``model=``. Its ``RoutingStrategy``
selects the candidates in preference order once per invocation, and the router routes across them:
it caches that order, uses the most preferred candidate, and advances when a model fails and no hook
has claimed the retry. Each candidate receives a fresh retry budget.

Fallback is cyclic. Within a round the router advances through candidates it has not yet tried, and
a successful call re-arms every other candidate, so a later failure restarts from the most preferred
candidate even if that one already failed earlier in the invocation. A degraded call therefore does
not pin the rest of the invocation to a less preferred model.

Candidates that failed during the invocation are demoted rather than excluded: fallback tries the
fewest-failed candidates first, so a model that keeps failing sinks below the healthy ones instead
of being re-probed with a fresh retry budget every round. It stays reachable, and a success clears
its record, so a recovered model returns to full preference.

A nested ``ModelRouter`` is one atomic position: its own strategy chooses its model, while the outer
router controls when to leave that position.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Union

from ..._middleware.stages import InvokeModelStage
from ...hooks.events import AfterInvocationEvent, AfterModelCallEvent
from ...hooks.registry import HookOrder
from ...plugins.plugin import Plugin
from ..model import Model
from .strategy import RoutingContext, RoutingStrategy

if TYPE_CHECKING:
    from ..._middleware.stages import InvokeModelContext
    from ...agent.agent import Agent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingCandidate:
    """A routing candidate: a model with an optional name and description.

    ``model`` may be a nested ``ModelRouter``; it is then one atomic position whose own strategy
    chooses the concrete model.
    """

    model: Model | ModelRouter
    name: str | None = None
    description: str | None = None


CandidateInput = Union[Model, "ModelRouter", RoutingCandidate]

_ROUTER_PLUGIN_NAME = "strands:model-router"
_ROUTING_KEY_PREFIX = "strands:model_routing"


@dataclass
class _RoutingState:
    """Per-invocation route execution state for one agent/router pair.

    ``tried_positions`` covers the current fallback round only; a successful call collapses it to
    the live position so the other candidates are eligible again. ``failure_counts`` spans the whole
    invocation and only demotes: candidates are tried fewest-failures first, so a long tool loop
    stops paying a full retry budget to rediscover the same dead model. A success clears that
    candidate's count, so a recovered model returns to full preference.
    """

    route: tuple[RoutingCandidate, ...]
    position: int
    model: Model
    tried_positions: set[int]
    failure_counts: dict[int, int] = field(default_factory=dict)


class FallbackStrategy:
    """Selects candidates in declaration order."""

    async def select(self, context: RoutingContext, **kwargs: Any) -> Sequence[RoutingCandidate]:
        """Return candidates in declaration order."""
        return context.candidates


class ModelRouter(Plugin):
    """A reusable set of candidate models routed in strategy-defined preference order."""

    def __init__(self, models: Sequence[CandidateInput], *, strategy: RoutingStrategy | None = None) -> None:
        """Initialize the router.

        Args:
            models: Candidates as a sequence. Each is a ``Model``, a nested ``ModelRouter``, or a
                ``RoutingCandidate`` carrying an optional name/description. The first candidate is
                the router's concrete default.
            strategy: Orders candidates by preference for each invocation, and may return only the
                ones it cares about. Defaults to ``FallbackStrategy``, which keeps declaration order.

        Raises:
            TypeError: If ``models`` is not a sequence, a candidate is not a ``Model`` or
                ``ModelRouter``, or ``strategy`` does not implement ``RoutingStrategy``.
            ValueError: If ``models`` is empty, candidate names collide, or any candidate is a
                stateful model.
        """
        super().__init__()
        if strategy is not None and not isinstance(strategy, RoutingStrategy):
            raise TypeError("strategy must implement RoutingStrategy (a select(context) method)")
        candidates = _normalize(models)
        if not candidates:
            raise ValueError("ModelRouter requires at least one candidate model")
        _reject_stateful(candidates)
        _reject_duplicates(candidates)
        self._candidates = candidates
        self._strategy: RoutingStrategy = strategy or FallbackStrategy()

    @property
    def name(self) -> str:
        """Stable plugin identifier."""
        return _ROUTER_PLUGIN_NAME

    @property
    def candidates(self) -> tuple[RoutingCandidate, ...]:
        """The normalized candidates, in declaration order."""
        return self._candidates

    @property
    def default_model(self) -> Model:
        """The first declared candidate resolved to a concrete model."""
        model = self._candidates[0].model
        if isinstance(model, ModelRouter):
            return model.default_model
        return model

    def init_agent(self, agent: Agent) -> None:
        """Register routing middleware and hooks; reject attachment through ``plugins=[...]``.

        Args:
            agent: The agent the router is attached to.

        Raises:
            ValueError: If the router was not attached through ``Agent(model=...)``.
        """
        if agent._model_router is not self:
            raise ValueError("ModelRouter must be passed through Agent(model=...), not plugins=[...]")

        agent._middleware_registry.add_middleware(InvokeModelStage.Input, self._selection_middleware())
        # Fallback must see whether ModelRetryStrategy (DEFAULT) already claimed the retry.
        agent.hooks.add_callback(AfterModelCallEvent, self._on_model_result, order=HookOrder.MODEL_ROUTING)
        # Teardown runs last so other end-of-invocation callbacks still observe the selection.
        agent.hooks.add_callback(AfterInvocationEvent, self._clear_state, order=HookOrder.SDK_LAST)

    async def _plan(self, context: RoutingContext) -> tuple[RoutingCandidate, ...]:
        """Order candidates by strategy preference, appending any it left out in declaration order.

        Completing the route keeps every candidate reachable by fallback, so a strategy may return
        only the candidates it cares about.
        """
        preferred: object = await self._strategy.select(context)
        # str/bytes/Mapping satisfy Sequence but are never a candidate order; naming a candidate is
        # the likeliest mistake, so report it as the type error it is.
        if isinstance(preferred, (str, bytes, Mapping)) or not isinstance(preferred, Sequence):
            raise TypeError(f"strategy.select must return a sequence of candidates; got {type(preferred).__name__}")

        configured_ids = {id(candidate) for candidate in context.candidates}
        route: list[RoutingCandidate] = []
        ranked_ids: set[int] = set()
        for candidate in preferred:
            if id(candidate) not in configured_ids:
                raise ValueError("strategy.select must return candidates from context.candidates")
            if id(candidate) not in ranked_ids:
                ranked_ids.add(id(candidate))
                route.append(candidate)

        route.extend(candidate for candidate in context.candidates if id(candidate) not in ranked_ids)
        return tuple(route)

    async def _resolve(self, candidate: RoutingCandidate, context: RoutingContext) -> Model:
        """Resolve a candidate to a concrete model, recursing into a nested router's selection."""
        model = candidate.model
        if isinstance(model, ModelRouter):
            return await model._select_model(replace(context, candidates=model.candidates))
        return model

    async def _select_model(self, context: RoutingContext) -> Model:
        """Resolve the strategy's most preferred candidate."""
        route = await self._plan(context)
        return await self._resolve(route[0], context)

    def _selection_middleware(self) -> Callable[[InvokeModelContext], Awaitable[InvokeModelContext]]:
        """Build an ``InvokeModelStage.Input`` handler that applies the per-invocation selection."""

        async def middleware(context: InvokeModelContext) -> InvokeModelContext:
            key = self._state_key(context.agent)
            state = _routing_state(context.invocation_state, key)
            if state is None:
                routing_context = self._routing_context(
                    context.messages, context.system_prompt, context.tool_specs, context.invocation_state
                )
                route = await self._plan(routing_context)
                state = _RoutingState(
                    route=route,
                    position=0,
                    model=await self._resolve(route[0], routing_context),
                    tried_positions={0},
                )
                context.invocation_state[key] = state
            context.model = state.model
            return context

        return middleware

    async def _on_model_result(self, event: AfterModelCallEvent) -> None:
        """Start a new fallback round on success or advance within the current round after a failure."""
        state = _routing_state(event.invocation_state, self._state_key(event.agent))
        if state is None:
            return
        if event.stop_response is not None:
            state.tried_positions = {state.position}
            state.failure_counts.pop(state.position, None)  # it works now; stop demoting it
            return
        if event.retry or event.exception is None:
            return

        state.failure_counts[state.position] = state.failure_counts.get(state.position, 0) + 1
        routing_context = self._agent_routing_context(event.agent, event.invocation_state)
        for next_position in self._advance_order(state):
            candidate = state.route[next_position]
            state.tried_positions.add(next_position)
            try:
                model = await self._resolve(candidate, routing_context)
            except Exception as error:
                # Preserve the model error and continue through the remaining candidates.
                logger.warning(
                    "candidate=<%s>, position=<%d>, error=<%s> | fallback resolution failed",
                    _candidate_label(candidate),
                    next_position,
                    error,
                )
                continue

            current = state.route[state.position]
            logger.info(
                "from_candidate=<%s>, to_candidate=<%s>, error=<%s> | model call failed, advancing candidate",
                _candidate_label(current),
                _candidate_label(candidate),
                type(event.exception).__name__,
            )
            state.model = model
            state.position = next_position
            event.agent._retry_strategy._reset_retry_state()
            event.retry = True
            return

    async def _clear_state(self, event: AfterInvocationEvent) -> None:
        """Drop this agent's routing state at the end of the invocation."""
        key = self._state_key(event.agent)
        if _routing_state(event.invocation_state, key) is not None:
            del event.invocation_state[key]

    def _advance_order(self, state: _RoutingState) -> list[int]:
        """Return untried positions, fewest failures first, then by preference."""
        return sorted(
            (position for position in range(len(state.route)) if position not in state.tried_positions),
            key=lambda position: (state.failure_counts.get(position, 0), position),
        )

    def _state_key(self, agent: object) -> str:
        """Scope routing state to one agent/router pair.

        One ``invocation_state`` can serve several agents (parallel Graph nodes) and one router can
        be attached to several agents, so neither identity alone is a sufficient key.
        """
        return f"{_ROUTING_KEY_PREFIX}:{id(agent):x}:{id(self):x}"

    def _agent_routing_context(self, agent: Any, invocation_state: Mapping[str, Any]) -> RoutingContext:
        """Build a ``RoutingContext`` from the agent, matching the shapes middleware passes."""
        system_prompt = (
            agent._system_prompt_content if agent._system_prompt_content is not None else agent.system_prompt
        )
        return self._routing_context(
            copy.deepcopy(agent.messages),
            copy.deepcopy(system_prompt),
            agent.tool_registry.get_all_tool_specs(),
            invocation_state,
        )

    def _routing_context(
        self, messages: Any, system_prompt: Any, tool_specs: Any, invocation_state: Mapping[str, Any]
    ) -> RoutingContext:
        """Build a ``RoutingContext`` over this router's candidates."""
        return RoutingContext(
            messages=messages,
            system_prompt=system_prompt,
            tool_specs=tool_specs,
            candidates=self._candidates,
            invocation_state=invocation_state,
        )


def _candidate_label(candidate: RoutingCandidate) -> str:
    """Return a stable human-readable label for logs."""
    return candidate.name or type(candidate.model).__name__


def _routing_state(invocation_state: Mapping[str, Any], key: str) -> _RoutingState | None:
    """Return the routing state stored under ``key``, ignoring any foreign value."""
    value = invocation_state.get(key)
    return value if isinstance(value, _RoutingState) else None


def _normalize(models: object) -> tuple[RoutingCandidate, ...]:
    """Coerce the input sequence into ``RoutingCandidate`` objects, validating candidate types."""
    if isinstance(models, (str, bytes, Mapping)) or not isinstance(models, Sequence):
        raise TypeError("models must be a sequence of candidates")
    return tuple(_as_candidate(item) for item in models)


def _as_candidate(item: CandidateInput) -> RoutingCandidate:
    """Wrap a candidate input in a ``RoutingCandidate``, validating its model type."""
    candidate = item if isinstance(item, RoutingCandidate) else RoutingCandidate(model=item)
    if not isinstance(candidate.model, (Model, ModelRouter)):
        raise TypeError(f"candidate must be a Model or ModelRouter; got {type(candidate.model).__name__}")
    return candidate


def _reject_stateful(candidates: tuple[RoutingCandidate, ...]) -> None:
    """Reject any stateful candidate model."""
    for candidate in candidates:
        if isinstance(candidate.model, Model) and candidate.model.stateful:
            raise ValueError(
                f"candidate=<{_candidate_label(candidate)}> is stateful; routing among stateful models is not supported"
            )


def _reject_duplicates(candidates: tuple[RoutingCandidate, ...]) -> None:
    """Reject repeated candidate instances or colliding names; repeated models are allowed."""
    seen_candidates: set[int] = set()
    seen_names: set[str] = set()
    for candidate in candidates:
        identity = id(candidate)
        if identity in seen_candidates:
            raise ValueError("duplicate RoutingCandidate instance")
        seen_candidates.add(identity)

        if candidate.name is None:
            continue
        if candidate.name in seen_names:
            raise ValueError(f"duplicate candidate name=<{candidate.name}>")
        seen_names.add(candidate.name)
