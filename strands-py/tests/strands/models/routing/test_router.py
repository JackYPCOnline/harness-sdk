"""Tests for ModelRouter core: candidate normalization, default resolution, guards."""

import dataclasses

import pytest

from strands import Agent, Plugin
from strands.models import BedrockModel
from strands.models.routing import ModelRouter, RoutingCandidate
from strands.models.routing.strategy import RoutingStrategy
from tests.fixtures.mocked_model_provider import MockedModelProvider


class _StubStrategy:
    """Minimal strategy; routing selection is exercised in later phases."""

    name = "stub"

    async def select(self, context):
        return context.candidates[0]


class StatefulModel(MockedModelProvider):
    @property
    def stateful(self):
        return True


def _model(text="hi"):
    return MockedModelProvider([{"role": "assistant", "content": [{"text": text}]}])


# --- RoutingCandidate ---


def test_routing_candidate_is_frozen_with_optional_metadata():
    bare = RoutingCandidate(model=_model())
    assert (bare.name, bare.description) == (None, None)

    named = RoutingCandidate(model=_model(), name="a", description="desc")
    with pytest.raises(dataclasses.FrozenInstanceError):
        named.name = "b"


# --- plugin identity ---


def test_router_is_a_plugin_with_stable_name():
    router = ModelRouter(models=[_model()], strategy=_StubStrategy())

    assert isinstance(router, Plugin)
    assert router.name == "strands:model-router"


def test_stub_strategy_satisfies_protocol():
    assert isinstance(_StubStrategy(), RoutingStrategy)


# --- default resolution (first candidate) ---


def test_default_model_is_first_candidate():
    m0, m1 = _model("0"), _model("1")
    router = ModelRouter(models=[m0, m1], strategy=_StubStrategy())

    assert router.default_model is m0


def test_default_model_uses_first_even_with_names():
    haiku = BedrockModel(model_id="haiku")
    opus = BedrockModel(model_id="opus")
    router = ModelRouter(
        models=[RoutingCandidate(model=haiku, name="cheap"), RoutingCandidate(model=opus, name="strong")],
        strategy=_StubStrategy(),
    )

    assert router.default_model is haiku


# --- candidates + naming ---


def test_bare_models_become_unnamed_candidates():
    m0, m1 = _model(), _model()
    router = ModelRouter(models=[m0, m1], strategy=_StubStrategy())

    assert [(c.model, c.name) for c in router.candidates] == [(m0, None), (m1, None)]


def test_routing_candidate_metadata_is_preserved():
    m = _model()
    router = ModelRouter(
        models=[RoutingCandidate(model=m, name="routine", description="simple tasks")],
        strategy=_StubStrategy(),
    )

    candidate = router.candidates[0]
    assert (candidate.model, candidate.name, candidate.description) == (m, "routine", "simple tasks")


def test_duplicate_candidate_names_raise():
    with pytest.raises(ValueError, match="duplicate"):
        ModelRouter(
            models=[RoutingCandidate(_model(), name="a"), RoutingCandidate(_model(), name="a")],
            strategy=_StubStrategy(),
        )


def test_repeated_unnamed_candidates_are_allowed():
    router = ModelRouter(models=[_model(), _model()], strategy=_StubStrategy())
    assert len(router.candidates) == 2


# --- shorthand + nesting resolution ---


def test_string_candidate_resolves_to_bedrock_model():
    router = ModelRouter(models=["my-model-id"], strategy=_StubStrategy())
    default = router.default_model

    assert isinstance(default, BedrockModel)
    assert default.config.get("model_id") == "my-model-id"


def test_nested_router_default_resolves_recursively():
    inner_model = _model()
    inner = ModelRouter(models=[inner_model], strategy=_StubStrategy())
    outer = ModelRouter(models=[inner, _model("x")], strategy=_StubStrategy())

    assert outer.default_model is inner_model


# --- guards ---


def test_empty_models_raises():
    with pytest.raises(ValueError, match="at least one"):
        ModelRouter(models=[], strategy=_StubStrategy())


def test_stateful_candidate_raises():
    with pytest.raises(ValueError, match="stateful"):
        ModelRouter(models=[StatefulModel([])], strategy=_StubStrategy())


def test_mapping_models_raises():
    with pytest.raises(TypeError, match="sequence of candidates"):
        ModelRouter(models={"cheap": _model()}, strategy=_StubStrategy())


def test_bare_string_models_raises():
    with pytest.raises(TypeError, match="sequence of candidates"):
        ModelRouter(models="my-model-id", strategy=_StubStrategy())


# --- agent integration ---


def test_agent_accepts_model_router_and_exposes_default():
    m = _model("routed")
    router = ModelRouter(models=[m], strategy=_StubStrategy())
    agent = Agent(model=router, callback_handler=None)

    assert agent.model is m
    assert agent._model_router is router


def test_agent_registers_router_as_plugin():
    router = ModelRouter(models=[_model()], strategy=_StubStrategy())
    agent = Agent(model=router, callback_handler=None)

    assert router.name in agent._plugin_registry._plugins


def test_agent_runs_with_router_using_first_candidate():
    router = ModelRouter(models=[_model("routed")], strategy=_StubStrategy())
    agent = Agent(model=router, callback_handler=None)

    result = agent("hello")

    assert result.message["content"][0]["text"] == "routed"
