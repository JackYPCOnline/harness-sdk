"""ModelRouter: a reusable, immutable set of candidate models with a routing strategy.

A router is a ``Plugin`` so an agent can accept it through ``model=``. Its ``RoutingStrategy``
selects the candidates in preference order once per invocation, and the router routes across them:
it caches that order, uses the most preferred candidate, and advances when a model fails and no hook
has claimed the retry. Each candidate receives a fresh retry budget, and a successful call re-arms
the remaining candidates. A nested ``ModelRouter`` is one atomic position: its own strategy chooses
its model, while the outer router controls when to leave that position.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
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
_ROUTING_KEY = "strands:model_routing"


@dataclass
class _RoutingState:
    """Per-invocation route execution state owned by one router."""

    router: ModelRouter
    route: tuple[RoutingCandidate, ...]
    position: int
    model: Model
    tried_positions: set[int]


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
            strategy: Selects every candidate in preference order for each invocation. Defaults to
                ``FallbackStrategy``, which preserves declaration order.

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
        # MODEL_ROUTING runs after default-order retry decisions and before SDK_LAST.
        agent.hooks.add_callback(AfterModelCallEvent, self._on_model_result, order=HookOrder.MODEL_ROUTING)
        agent.hooks.add_callback(AfterInvocationEvent, self._clear_state, order=HookOrder.MODEL_ROUTING)

    async def _plan(self, context: RoutingContext) -> tuple[RoutingCandidate, ...]:
        """Build and validate the strategy's preference order."""
        result = await self._strategy.select(context)
        if not isinstance(result, Sequence):
            raise TypeError("strategy.select must return a sequence of candidates")

        route = tuple(result)
        configured_ids = {id(candidate) for candidate in context.candidates}
        if (
            len(route) != len(context.candidates)
            or len({id(candidate) for candidate in route}) != len(route)
            or any(id(candidate) not in configured_ids for candidate in route)
        ):
            raise ValueError("strategy.select must return every candidate exactly once")
        return route

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
            state = _owned_state(context.invocation_state.get(_ROUTING_KEY), self)
            if state is None:
                routing_context = self._routing_context(
                    context.messages, context.system_prompt, context.tool_specs, context.invocation_state
                )
                route = await self._plan(routing_context)
                state = _RoutingState(
                    router=self,
                    route=route,
                    position=0,
                    model=await self._resolve(route[0], routing_context),
                    tried_positions={0},
                )
                context.invocation_state[_ROUTING_KEY] = state
            context.model = state.model
            return context

        return middleware

    async def _on_model_result(self, event: AfterModelCallEvent) -> None:
        """Re-arm the remaining candidates on success or advance after an unretried failure."""
        state = _owned_state(event.invocation_state.get(_ROUTING_KEY), self)
        if state is None:
            return
        if event.stop_response is not None:
            state.tried_positions = {state.position}
            return
        if event.retry or event.exception is None:
            return

        routing_context = self._routing_context(
            event.agent.messages,
            event.agent.system_prompt,
            event.agent.tool_registry.get_all_tool_specs(),
            event.invocation_state,
        )
        for next_position, candidate in enumerate(state.route):
            if next_position in state.tried_positions:
                continue
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
        """Drop this router's state at the end of the invocation."""
        if _owned_state(event.invocation_state.get(_ROUTING_KEY), self) is not None:
            del event.invocation_state[_ROUTING_KEY]

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


def _owned_state(value: object, router: ModelRouter) -> _RoutingState | None:
    """Return the routing state if it belongs to this router."""
    if isinstance(value, _RoutingState) and value.router is router:
        return value
    return None


def _normalize(models: object) -> tuple[RoutingCandidate, ...]:
    """Coerce the input sequence into ``RoutingCandidate`` objects, validating candidate types."""
    if isinstance(models, (str, bytes, Mapping)) or not isinstance(models, Sequence):
        raise TypeError("models must be a sequence of candidates")
    return tuple(_as_candidate(item) for item in models)


def _as_candidate(item: CandidateInput) -> RoutingCandidate:
    """Wrap a candidate input in a ``RoutingCandidate``, validating its model type."""
    if isinstance(item, RoutingCandidate):
        _validate_candidate_model(item.model)
        return item
    return RoutingCandidate(model=_validate_candidate_model(item))


def _validate_candidate_model(model: object) -> Model | ModelRouter:
    """Return the model if it is a ``Model`` or ``ModelRouter``; reject other types."""
    if isinstance(model, (Model, ModelRouter)):
        return model
    raise TypeError(f"candidate must be a Model or ModelRouter; got {type(model).__name__}")


def _reject_stateful(candidates: tuple[RoutingCandidate, ...]) -> None:
    """Reject any stateful candidate model."""
    for candidate in candidates:
        if isinstance(candidate.model, Model) and candidate.model.stateful:
            label = candidate.name or type(candidate.model).__name__
            raise ValueError(f"candidate=<{label}> is stateful; routing among stateful models is not supported")


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
