"""Tests for ModelRouter core: candidate validation, strategy selection, guards."""

import asyncio
import contextlib
import types

import pytest

from strands import Agent, Plugin
from strands.event_loop._retry import ModelRetryStrategy
from strands.models import BedrockModel
from strands.models.routing import (
    FallbackStrategy,
    ModelRouter,
    RoutingCandidate,
    RoutingContext,
    RoutingStrategy,
)
from strands.models.routing.router import _RoutingState
from strands.multiagent import GraphBuilder
from strands.types.exceptions import ModelThrottledException
from tests.fixtures.mocked_model_provider import MockedModelProvider


class StatefulModel(MockedModelProvider):
    @property
    def stateful(self):
        return True


def _model(text="hi"):
    return MockedModelProvider([{"role": "assistant", "content": [{"text": text}]}])


class _PreferByName:
    """Strategy that puts named candidates first and counts its calls."""

    def __init__(self, *names):
        self.names = names
        self.calls = 0

    async def select(self, context):
        self.calls += 1
        by_name = {candidate.name: candidate for candidate in context.candidates}
        prioritized = tuple(by_name[name] for name in self.names)
        prioritized_ids = {id(candidate) for candidate in prioritized}
        return prioritized + tuple(c for c in context.candidates if id(c) not in prioritized_ids)


def _routing_context(candidates, invocation_state=None):
    return RoutingContext(
        messages=[],
        system_prompt=None,
        tool_specs=[],
        candidates=candidates,
        invocation_state=invocation_state if invocation_state is not None else {},
    )


_TEST_AGENT = object()


def _invoke_context(invocation_state, model, agent=None):
    return types.SimpleNamespace(
        agent=agent if agent is not None else _TEST_AGENT,
        messages=[],
        system_prompt=None,
        tool_specs=[],
        invocation_state=invocation_state,
        model=model,
    )


# --- plugin identity ---


def test_router_is_a_plugin_with_stable_name():
    router = ModelRouter(models=[_model()])

    assert isinstance(router, Plugin)
    assert router.name == "strands:model-router"


# --- candidates + metadata ---


def test_routing_candidate_metadata_is_preserved():
    m = _model()
    router = ModelRouter(models=[RoutingCandidate(model=m, name="routine", description="simple tasks")])

    candidate = router.candidates[0]
    assert (candidate.model, candidate.name, candidate.description) == (m, "routine", "simple tasks")


def test_repeated_model_object_is_allowed():
    m = _model()
    router = ModelRouter(models=[m, m])

    assert router.default_model is m


def test_bedrock_model_object_is_a_valid_candidate():
    haiku = BedrockModel(model_id="haiku")
    router = ModelRouter(models=[haiku, BedrockModel(model_id="opus")])

    assert router.default_model is haiku


# --- default resolution (first candidate) ---


def test_default_model_is_first_candidate():
    m0, m1 = _model("0"), _model("1")
    router = ModelRouter(models=[m0, m1])

    assert router.default_model is m0


def test_nested_router_default_resolves_recursively():
    inner_model = _model()
    inner = ModelRouter(models=[inner_model])
    outer = ModelRouter(models=[inner, _model("x")])

    assert outer.default_model is inner_model


# --- strategy selection ---


@pytest.mark.asyncio
async def test_fallback_strategy_selects_in_declaration_order():
    router = ModelRouter(models=[_model(), _model()])

    selected = await FallbackStrategy().select(_routing_context(router.candidates))

    assert tuple(selected) == router.candidates
    assert await router._select_model(_routing_context(router.candidates)) is router.candidates[0].model


@pytest.mark.asyncio
async def test_custom_strategy_prefers_named_candidate():
    fast, smart = _model(), _model()
    router = ModelRouter(
        models=[RoutingCandidate(fast, name="fast"), RoutingCandidate(smart, name="smart")],
        strategy=_PreferByName("smart"),
    )

    assert await router._select_model(_routing_context(router.candidates)) is smart


@pytest.mark.asyncio
async def test_selection_recurses_into_nested_router_strategy():
    inner_fast, inner_smart = _model(), _model()
    inner = ModelRouter(
        models=[RoutingCandidate(inner_fast, name="if"), RoutingCandidate(inner_smart, name="is")],
        strategy=_PreferByName("is"),
    )
    outer = ModelRouter(
        models=[_model(), RoutingCandidate(inner, name="inner")],
        strategy=_PreferByName("inner"),
    )

    assert await outer._select_model(_routing_context(outer.candidates)) is inner_smart


def test_non_strategy_raises():
    with pytest.raises(TypeError, match="RoutingStrategy"):
        ModelRouter(models=[_model()], strategy=object())


def test_routing_strategy_protocol_is_runtime_checkable():
    assert isinstance(_PreferByName("x"), RoutingStrategy)
    assert not isinstance(object(), RoutingStrategy)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("returns", "exc", "match"),
    [
        (lambda c: c[0], TypeError, "sequence of candidates; got RoutingCandidate"),
        # A judge naming a candidate is the likeliest mistake, and str satisfies Sequence.
        (lambda c: "cheap", TypeError, "sequence of candidates; got str"),
        (lambda c: {"cheap": 1}, TypeError, "sequence of candidates; got dict"),
        (lambda c: [c[0], RoutingCandidate(_model())], ValueError, "from context.candidates"),
    ],
    ids=["single-candidate", "candidate-name-string", "mapping", "foreign-candidate"],
)
async def test_strategy_selection_rejects_unusable_results(returns, exc, match):
    class _InvalidSelection:
        async def select(self, context):
            return returns(context.candidates)

    router = ModelRouter(models=[_model(), _model()], strategy=_InvalidSelection())

    with pytest.raises(exc, match=match):
        await router._select_model(_routing_context(router.candidates))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("returned", "expected_order"),
    [
        (lambda c: [], [0, 1, 2]),
        (lambda c: [c[2]], [2, 0, 1]),
        (lambda c: [c[2], c[2], c[1]], [2, 1, 0]),
        (lambda c: [c[1], c[0], c[2]], [1, 0, 2]),
    ],
    ids=["empty-keeps-declaration-order", "subset-is-completed", "duplicates-collapse", "full-permutation"],
)
async def test_route_always_covers_every_candidate(returned, expected_order):
    # Fallback must be able to reach every candidate, so the router completes whatever the strategy
    # returns rather than rejecting a partial preference.
    class _PartialSelection:
        async def select(self, context):
            return returned(context.candidates)

    router = ModelRouter(models=[_model(), _model(), _model()], strategy=_PartialSelection())

    route = await router._plan(_routing_context(router.candidates))

    assert route == tuple(router.candidates[index] for index in expected_order)


# --- selection middleware ---


@pytest.mark.asyncio
async def test_selection_middleware_sets_model_and_caches_per_invocation_state():
    fast, smart = _model(), _model()
    strategy = _PreferByName("smart")
    router = ModelRouter(
        models=[RoutingCandidate(fast, name="fast"), RoutingCandidate(smart, name="smart")], strategy=strategy
    )
    middleware = router._selection_middleware()

    # Same invocation_state: selects once, then reuses the cached model.
    state: dict = {}
    assert (await middleware(_invoke_context(state, model=fast))).model is smart
    assert (await middleware(_invoke_context(state, model=fast))).model is smart
    assert strategy.calls == 1

    # A fresh invocation_state selects again.
    assert (await middleware(_invoke_context({}, model=fast))).model is smart
    assert strategy.calls == 2


# --- construction guards ---


@pytest.mark.parametrize(
    ("make_models", "exc", "match"),
    [
        (lambda: [], ValueError, "at least one"),
        (lambda: "my-model-id", TypeError, "sequence of candidates"),
        (lambda: {"cheap": _model()}, TypeError, "sequence of candidates"),
        (lambda: ["my-model-id"], TypeError, "candidate must be"),
        (lambda: [object()], TypeError, "candidate must be"),
        (lambda: [StatefulModel([])], ValueError, r"StatefulModel.*stateful"),
        (lambda: [RoutingCandidate(StatefulModel([]), name="vip")], ValueError, r"vip.*stateful"),
        (
            lambda: [RoutingCandidate(_model(), name="a"), RoutingCandidate(_model(), name="a")],
            ValueError,
            "duplicate candidate name",
        ),
        (lambda: [RoutingCandidate(_model())] * 2, ValueError, "duplicate RoutingCandidate instance"),
    ],
    ids=[
        "empty",
        "bare-string",
        "mapping",
        "string-candidate",
        "invalid-object",
        "stateful",
        "stateful-uses-name-in-error",
        "duplicate-name",
        "duplicate-instance",
    ],
)
def test_construction_rejects_invalid_input(make_models, exc, match):
    with pytest.raises(exc, match=match):
        ModelRouter(models=make_models())


# --- agent integration ---


def test_agent_accepts_model_router_and_exposes_default():
    m = _model("routed")
    router = ModelRouter(models=[m])
    agent = Agent(model=router, callback_handler=None)

    assert agent.model is m
    assert agent._model_router is router


def test_agent_registers_router_as_plugin():
    router = ModelRouter(models=[_model()])
    agent = Agent(model=router, callback_handler=None)

    assert router.name in agent._plugin_registry._plugins


def test_router_via_plugins_is_rejected():
    router = ModelRouter(models=[_model()])
    with pytest.raises(ValueError, match=r"model=.*not plugins"):
        Agent(plugins=[router], callback_handler=None)


def test_agent_runs_with_default_first_candidate():
    router = ModelRouter(models=[_model("routed")])
    agent = Agent(model=router, callback_handler=None)

    result = agent("hello")

    assert result.message["content"][0]["text"] == "routed"


def test_agent_routes_to_strategy_selected_candidate():
    fast = _model("fast-says")
    smart = _model("smart-says")
    router = ModelRouter(
        models=[RoutingCandidate(fast, name="fast"), RoutingCandidate(smart, name="smart")],
        strategy=_PreferByName("smart"),
    )
    agent = Agent(model=router, callback_handler=None)

    assert agent.model is fast  # default is still the first candidate

    result = agent("hello")

    assert result.message["content"][0]["text"] == "smart-says"  # strategy overrode the per-call model


class _ModelProbe(Plugin):
    """Plugin that records the ``context.model`` seen by a downstream Input middleware."""

    name = "test:model-probe"

    def __init__(self):
        super().__init__()
        self.seen = None

    def init_agent(self, agent):
        from strands._middleware.stages import InvokeModelStage

        def record(context):
            self.seen = context.model
            return context

        agent._middleware_registry.add_middleware(InvokeModelStage.Input, record)


def test_routing_runs_before_other_input_middleware():
    fast = _model("f")
    smart = _model("s")
    router = ModelRouter(
        models=[RoutingCandidate(fast, name="fast"), RoutingCandidate(smart, name="smart")],
        strategy=_PreferByName("smart"),
    )
    probe = _ModelProbe()
    agent = Agent(model=router, plugins=[probe], callback_handler=None)

    agent("hello")

    assert probe.seen is smart  # routing set the per-call model before the probe middleware ran


# --- ordered fallback ---


class _FailingModel(MockedModelProvider):
    """A model whose stream always raises the given exception."""

    def __init__(self, exception):
        super().__init__([{"role": "assistant", "content": [{"text": "unused"}]}])
        self._exception = exception

    async def stream(self, *args, **kwargs):
        raise self._exception
        yield  # pragma: no cover - marks this an async generator


class _FlakyModel(MockedModelProvider):
    """A model that raises a throttling exception a set number of times, then streams normally."""

    def __init__(self, failures, text):
        super().__init__([{"role": "assistant", "content": [{"text": text}]}])
        self._remaining_failures = failures

    async def stream(self, *args, **kwargs):
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise ModelThrottledException("flaky")
        async for event in super().stream(*args, **kwargs):
            yield event


class _RaisingStrategy:
    """A strategy whose select always raises, to exercise resolution error containment."""

    async def select(self, context, **kwargs):
        raise RuntimeError("strategy boom")


def _agent_stub():
    """A minimal stand-in for the fields the fallback hook reads off the agent."""
    return types.SimpleNamespace(
        messages=[],
        system_prompt=None,
        _system_prompt_content=None,
        tool_registry=types.SimpleNamespace(get_all_tool_specs=lambda: []),
        _retry_strategy=ModelRetryStrategy(max_attempts=1),
    )


@pytest.mark.parametrize(
    "exception",
    [ModelThrottledException("throttled"), ValueError("boom")],
    ids=["retryable", "non-retryable"],
)
def test_fallback_advances_past_a_failing_candidate(exception):
    good = _model("recovered")
    router = ModelRouter(models=[_FailingModel(exception), good])
    agent = Agent(model=router, retry_strategy=None, callback_handler=None)

    result = agent("hello")

    assert result.message["content"][0]["text"] == "recovered"


def test_strategy_controls_both_initial_choice_and_fallback_order():
    declared_first = _model("declared-first")
    prioritized = _FailingModel(ValueError("priority failed"))
    fallback = _model("strategy-fallback")
    router = ModelRouter(
        models=[
            RoutingCandidate(declared_first, name="declared-first"),
            RoutingCandidate(prioritized, name="priority"),
            RoutingCandidate(fallback, name="fallback"),
        ],
        strategy=_PreferByName("priority", "fallback"),
    )
    agent = Agent(model=router, retry_strategy=None, callback_handler=None)

    result = agent("hello")

    assert result.message["content"][0]["text"] == "strategy-fallback"


def test_fallback_exhausts_all_candidates_then_raises():
    router = ModelRouter(
        models=[_FailingModel(ModelThrottledException("a")), _FailingModel(ModelThrottledException("b"))]
    )
    agent = Agent(model=router, retry_strategy=None, callback_handler=None)

    with pytest.raises(ModelThrottledException):
        agent("hello")


@pytest.mark.asyncio
async def test_advance_is_noop_without_routing_state():
    router = ModelRouter(models=[_model(), _model()])
    event = types.SimpleNamespace(
        retry=False, stop_response=None, exception=ValueError("x"), invocation_state={}, agent=None
    )

    await router._on_model_result(event)

    assert event.retry is False  # no selection was cached, so there is nothing to advance


def test_fallback_resets_retry_budget_so_next_candidate_gets_fresh_retries():
    # First candidate always fails; second needs two retries before it succeeds. This only passes if
    # advancing resets the retry budget so the second candidate gets its own attempts.
    first = _FailingModel(ModelThrottledException("down"))
    second = _FlakyModel(failures=2, text="recovered")
    router = ModelRouter(models=[first, second])
    retry_strategy = ModelRetryStrategy(max_attempts=3, initial_delay=0, max_delay=0)
    agent = Agent(model=router, retry_strategy=retry_strategy, callback_handler=None)

    result = agent("hello")

    assert result.message["content"][0]["text"] == "recovered"


# --- state scoping / lifecycle ---


@pytest.mark.asyncio
async def test_two_routers_on_one_agent_keep_separate_state():
    m_a, m_b = _model(), _model()
    router_a = ModelRouter(models=[m_a])
    router_b = ModelRouter(models=[m_b])
    shared: dict = {}

    context_a = _invoke_context(shared, model=None)
    await router_a._selection_middleware()(context_a)
    context_b = _invoke_context(shared, model=None)
    await router_b._selection_middleware()(context_b)

    assert (context_a.model, context_b.model) == (m_a, m_b)
    # Each router owns its own slot, so router_b's selection did not evict router_a's.
    assert (await router_a._selection_middleware()(_invoke_context(shared, model=None))).model is m_a


@pytest.mark.asyncio
async def test_two_agents_sharing_one_router_and_state_dict_route_independently():
    # Graph hands one invocation_state to every node, so a shared router must not let one agent
    # run on another agent's cached selection.
    fast, smart = _model(), _model()
    router = ModelRouter(
        models=[RoutingCandidate(fast, name="fast"), RoutingCandidate(smart, name="smart")],
        strategy=_PreferByName("smart"),
    )
    shared: dict = {}
    agent_one, agent_two = object(), object()

    context_one = _invoke_context(shared, model=None, agent=agent_one)
    await router._selection_middleware()(context_one)
    context_two = _invoke_context(shared, model=None, agent=agent_two)
    await router._selection_middleware()(context_two)

    assert (context_one.model, context_two.model) == (smart, smart)
    # Two distinct slots, so neither agent can advance or clear the other's route.
    assert len([key for key in shared if key.startswith("strands:model_routing")]) == 2

    await router._clear_state(types.SimpleNamespace(invocation_state=shared, agent=agent_one))
    assert [key for key in shared if key.startswith("strands:model_routing")] == [router._state_key(agent_two)]


@pytest.mark.asyncio
async def test_clear_state_removes_only_this_agents_routing_state():
    router = ModelRouter(models=[_model()])
    mine, theirs = object(), object()
    invocation_state = {
        router._state_key(mine): _RoutingState(
            route=router.candidates, position=0, model=router.default_model, tried_positions={0}
        ),
        router._state_key(theirs): _RoutingState(
            route=router.candidates, position=0, model=router.default_model, tried_positions={0}
        ),
    }

    await router._clear_state(types.SimpleNamespace(invocation_state=invocation_state, agent=mine))

    assert list(invocation_state) == [router._state_key(theirs)]

    # An agent that never selected has nothing to clear.
    await router._clear_state(types.SimpleNamespace(invocation_state=invocation_state, agent=object()))
    assert list(invocation_state) == [router._state_key(theirs)]


def test_router_does_not_clobber_caller_invocation_state():
    router = ModelRouter(models=[_model("ok")])
    agent = Agent(model=router, callback_handler=None)
    state = {"model_routing": "caller-owned", "keep": 1}

    agent("hi", invocation_state=state)

    assert state["model_routing"] == "caller-owned"
    assert state["keep"] == 1
    # Routing state is cleared at the end of the invocation.
    assert [key for key in state if key.startswith("strands:model_routing")] == []


@pytest.mark.asyncio
async def test_successful_call_rearms_the_fallback_chain():
    router = ModelRouter(models=[_model(), _model(), _model()])
    state = _RoutingState(
        route=router.candidates,
        position=1,
        model=router.candidates[1].model,
        tried_positions={0, 1},
    )
    event = types.SimpleNamespace(
        retry=False,
        stop_response=object(),
        exception=None,
        invocation_state={router._state_key(None): state},
        agent=None,
    )

    await router._on_model_result(event)

    assert state.tried_positions == {1}


@pytest.mark.asyncio
async def test_fallback_cycles_back_to_a_failed_candidate_in_the_next_round():
    # Fallback is cyclic: a candidate that failed earlier in the invocation becomes eligible again
    # after any successful call, so the route restarts from the most preferred candidate.
    router = ModelRouter(models=[_model("first"), _model("second")])
    agent = _agent_stub()
    state = _RoutingState(
        route=router.candidates,
        position=0,
        model=router.candidates[0].model,
        tried_positions={0},
    )
    invocation_state = {router._state_key(agent): state}

    def failed_call():
        return types.SimpleNamespace(
            retry=False,
            stop_response=None,
            exception=ValueError("down"),
            invocation_state=invocation_state,
            agent=agent,
        )

    def successful_call():
        return types.SimpleNamespace(
            retry=False,
            stop_response=object(),
            exception=None,
            invocation_state=invocation_state,
            agent=agent,
        )

    await router._on_model_result(failed_call())
    assert (state.position, state.model) == (1, router.candidates[1].model)

    await router._on_model_result(successful_call())

    # Round two: the current candidate fails and the router returns to the one that failed first.
    event = failed_call()
    await router._on_model_result(event)

    assert (state.position, state.model) == (0, router.candidates[0].model)
    assert event.retry is True


@pytest.mark.asyncio
async def test_repeatedly_failing_candidate_is_demoted_below_healthy_ones():
    # Cyclic re-arm must not keep paying a retry budget to rediscover a hard-down model, so
    # fallback prefers the fewest-failed candidate rather than raw declaration order.
    dead, live, spare = _model("dead"), _model("live"), _model("spare")
    router = ModelRouter(
        models=[
            RoutingCandidate(dead, name="dead"),
            RoutingCandidate(live, name="live"),
            RoutingCandidate(spare, name="spare"),
        ]
    )
    agent = _agent_stub()
    state = _RoutingState(route=router.candidates, position=0, model=dead, tried_positions={0})
    invocation_state = {router._state_key(agent): state}

    def call(succeeded):
        return types.SimpleNamespace(
            retry=False,
            stop_response=object() if succeeded else None,
            exception=None if succeeded else ValueError("down"),
            invocation_state=invocation_state,
            agent=agent,
        )

    await router._on_model_result(call(succeeded=False))  # dead fails, advance to live
    assert state.position == 1
    await router._on_model_result(call(succeeded=True))  # live succeeds, re-arming the route

    # Round two: live blips. Declaration order would return to dead; demotion picks spare instead.
    await router._on_model_result(call(succeeded=False))

    assert state.position == 2
    assert state.model is spare

    # A candidate that succeeds again loses its demotion and regains full preference.
    await router._on_model_result(call(succeeded=True))
    assert state.failure_counts == {0: 1, 1: 1}


@pytest.mark.asyncio
async def test_fallback_resolution_error_is_contained():
    router = ModelRouter(models=[_model(), RoutingCandidate(ModelRouter([_model()], strategy=_RaisingStrategy()))])
    agent = _agent_stub()
    state = _RoutingState(
        route=router.candidates,
        position=0,
        model=router.default_model,
        tried_positions={0},
    )
    event = types.SimpleNamespace(
        retry=False,
        stop_response=None,
        exception=ValueError("original model error"),
        invocation_state={router._state_key(agent): state},
        agent=agent,
    )

    await router._on_model_result(event)

    assert event.retry is False  # a failed advance degrades to "no fallback" instead of crashing


def test_fallback_skips_an_unresolvable_candidate_and_reaches_a_healthy_one():
    # [0] fails on call, [1] is a nested router whose strategy raises during resolution, [2] is
    # healthy. A resolution failure on [1] must skip to [2], not abandon the chain.
    unresolvable = RoutingCandidate(ModelRouter([_model()], strategy=_RaisingStrategy()))
    router = ModelRouter(models=[_FailingModel(ValueError("primary down")), unresolvable, _model("healthy")])
    agent = Agent(model=router, retry_strategy=None, callback_handler=None)

    result = agent("hello")

    assert result.message["content"][0]["text"] == "healthy"


def test_nested_router_is_one_atomic_fallback_slot():
    inner = ModelRouter(models=[_FailingModel(ValueError("inner down")), _model("inner-second")])
    router = ModelRouter(models=[inner, _model("outer-other")])
    agent = Agent(model=router, retry_strategy=None, callback_handler=None)

    result = agent("hello")

    # The nested router's first pick fails; the outer router falls over to its own next candidate
    # rather than trying the nested router's second model.
    assert result.message["content"][0]["text"] == "outer-other"


class _RendezvousModel(MockedModelProvider):
    """Waits until every node has reached its model call, so the invocations genuinely overlap.

    Without this, Graph nodes finish one at a time and ``_clear_state`` removes the first node's
    state before the second selects, hiding cross-node state bleed.
    """

    def __init__(self, text, rendezvous, participants):
        super().__init__([{"role": "assistant", "content": [{"text": text}]} for _ in range(4)])
        self._rendezvous = rendezvous
        self._participants = participants
        self.calls = 0

    async def stream(self, *args, **kwargs):
        self.calls += 1
        self._rendezvous["arrived"] += 1
        if self._rendezvous["arrived"] >= self._participants:
            self._rendezvous["gate"].set()
        with contextlib.suppress(asyncio.TimeoutError):
            # A leak means one model is never reached, so the gate must not block forever.
            await asyncio.wait_for(self._rendezvous["gate"].wait(), timeout=2)
        async for event in super().stream(*args, **kwargs):
            yield event


@pytest.mark.asyncio
async def test_parallel_graph_nodes_sharing_one_router_route_independently():
    # Graph hands one invocation_state to every node, so a router attached to two agents must scope
    # state by node as well: router identity alone lets one node run on another's selection.
    rendezvous = {"arrived": 0, "gate": asyncio.Event()}
    fast = _RendezvousModel("fast-done", rendezvous, participants=2)
    smart = _RendezvousModel("smart-done", rendezvous, participants=2)

    class _BySystemPrompt:
        """Routes on the agent's own system prompt, never on invocation_state["agent"]."""

        def __init__(self):
            self.calls = 0

        async def select(self, context):
            self.calls += 1
            prompt = context.system_prompt or ""
            text = prompt if isinstance(prompt, str) else " ".join(b.get("text", "") for b in prompt)
            wanted = "smart" if "smart" in text else "fast"
            return [candidate for candidate in context.candidates if candidate.name == wanted]

    strategy = _BySystemPrompt()
    router = ModelRouter(
        models=[RoutingCandidate(fast, name="fast"), RoutingCandidate(smart, name="smart")],
        strategy=strategy,
    )
    builder = GraphBuilder()
    builder.add_node(Agent(model=router, system_prompt="be fast", callback_handler=None), "fast_node")
    builder.add_node(Agent(model=router, system_prompt="be smart", callback_handler=None), "smart_node")

    result = await builder.build().invoke_async("go")

    assert strategy.calls == 2  # both nodes consulted the strategy
    assert (fast.calls, smart.calls) == (1, 1)  # neither node ran on the other's model
    texts = {node_id: node.result.message["content"][0]["text"] for node_id, node in result.results.items()}
    assert texts == {"fast_node": "fast-done", "smart_node": "smart-done"}
