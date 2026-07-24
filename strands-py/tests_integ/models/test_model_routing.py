"""End-to-end integration tests for model routing.

Validates that an ``Agent`` accepts a ``ModelRouter`` over real Bedrock models, exposes the
first candidate as ``agent.model``, and completes a real invocation. Per-call selection
lands in a later phase; here the router resolves to its first (default) candidate, so these
tests exercise the Agent -> router -> InvokeModelStage -> concrete model wiring end to end.
"""

import pytest

from strands import Agent
from strands.models import BedrockModel
from strands.models.routing import ModelRouter, RoutingCandidate

_HAIKU_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


class _FirstCandidateStrategy:
    """Selects the first candidate. Stand-in until built-in strategies land."""

    name = "first-candidate"

    async def select(self, context):
        return context.candidates[0]


@pytest.fixture
def haiku():
    return BedrockModel(model_id=_HAIKU_MODEL_ID)


@pytest.fixture
def default_model():
    return BedrockModel()


def test_router_agent_completes_invocation_using_first_candidate(haiku, default_model):
    router = ModelRouter(models=[haiku, default_model], strategy=_FirstCandidateStrategy())
    agent = Agent(model=router, load_tools_from_directory=False)

    # agent.model exposes the router's first candidate as the concrete default.
    assert agent.model is haiku

    result = agent("What is 2 + 2? Reply with just the number.")

    assert "4" in result.message["content"][0]["text"]


def test_router_agent_runs_with_named_candidates(haiku, default_model):
    router = ModelRouter(
        models=[
            RoutingCandidate(model=haiku, name="routine", description="Simple, direct questions."),
            RoutingCandidate(model=default_model, name="complex", description="Harder reasoning."),
        ],
        strategy=_FirstCandidateStrategy(),
    )
    agent = Agent(model=router, load_tools_from_directory=False)

    result = agent("Name the capital of France in one word.")

    assert "paris" in result.message["content"][0]["text"].lower()


def test_router_agent_resolves_nested_router_end_to_end(haiku, default_model):
    inner = ModelRouter(models=[haiku], strategy=_FirstCandidateStrategy())
    outer = ModelRouter(models=[inner, default_model], strategy=_FirstCandidateStrategy())
    agent = Agent(model=outer, load_tools_from_directory=False)

    assert agent.model is haiku

    result = agent("Reply with the word: ok")

    assert "ok" in result.message["content"][0]["text"].lower()
