"""Tests for the public middleware surface: agent.add_middleware() and context .replace()."""

import threading

import pytest

import strands.middleware as middleware_pkg
from strands import Agent
from strands.interrupt import _InterruptState
from strands.middleware import (
    ExecuteToolContext,
    InvokeModelContext,
    InvokeModelStage,
    MiddlewareResult,
)
from strands.types._events import ModelStopReason
from tests.fixtures.mocked_model_provider import MockedModelProvider


@pytest.fixture
def model():
    return MockedModelProvider([{"role": "assistant", "content": [{"text": "Hello!"}]}])


@pytest.fixture
def agent(model):
    return Agent(model=model, callback_handler=None)


# --- agent.add_middleware() end-to-end ---


def test_add_middleware_wrap_runs_through_public_api(agent):
    seen = []

    async def capture(context, next_fn):
        seen.append(context)
        async for event in next_fn(context):
            yield event

    agent.add_middleware(InvokeModelStage, capture)
    result = agent("test")

    assert result.message["content"][0]["text"] == "Hello!"
    assert len(seen) == 1
    assert isinstance(seen[0], InvokeModelContext)


def test_add_middleware_input_transforms_context(agent):
    seen_prompt = None

    def inject(context):
        return context.replace(system_prompt="Injected")

    async def capture(context, next_fn):
        nonlocal seen_prompt
        seen_prompt = context.system_prompt
        async for event in next_fn(context):
            yield event

    agent.add_middleware(InvokeModelStage.Input, inject)
    agent.add_middleware(InvokeModelStage, capture)
    agent("test")

    assert seen_prompt == "Injected"


def test_add_middleware_output_transforms_result(agent):
    def rewrite(result: MiddlewareResult[ModelStopReason]) -> MiddlewareResult[ModelStopReason]:
        stop_reason, _message, usage, metrics = result.value["stop"]
        return result.replace(
            value=ModelStopReason(
                stop_reason=stop_reason,
                message={"role": "assistant", "content": [{"text": "rewritten"}]},
                usage=usage,
                metrics=metrics,
            )
        )

    agent.add_middleware(InvokeModelStage.Output, rewrite)
    result = agent("test")

    assert result.message["content"][0]["text"] == "rewritten"


def test_add_middleware_returns_none(agent):
    """Registration is one-way — no cleanup handle, matching the Python hook system."""

    async def passthrough(context, next_fn):
        async for event in next_fn(context):
            yield event

    assert agent.add_middleware(InvokeModelStage, passthrough) is None


# --- public surface ---


def test_agent_stream_stage_is_not_public():
    assert "AgentStreamStage" not in middleware_pkg.__all__
    assert "AgentStreamContext" not in middleware_pkg.__all__


# --- InvokeModelContext.replace() ---


@pytest.fixture
def invoke_context(agent):
    return InvokeModelContext(
        agent=agent,
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        system_prompt="original",
        tool_specs=[],
        tool_choice=None,
        invocation_state={},
        model=agent.model,
    )


def test_invoke_replace_changes_only_named_field(invoke_context):
    modified = invoke_context.replace(system_prompt="new")

    assert modified.system_prompt == "new"
    assert modified.messages is invoke_context.messages
    assert modified.model is invoke_context.model


def test_invoke_replace_leaves_original_unchanged(invoke_context):
    invoke_context.replace(system_prompt="new")

    assert invoke_context.system_prompt == "original"


def test_invoke_replace_honors_explicit_none(invoke_context):
    """An explicit None must override the current value, not be treated as 'unspecified'."""
    context = invoke_context.replace(system_prompt="set")
    modified = context.replace(system_prompt=None)

    assert modified.system_prompt is None


def test_invoke_replace_no_args_is_equivalent_copy(invoke_context):
    modified = invoke_context.replace()

    assert modified is not invoke_context
    assert modified.system_prompt == invoke_context.system_prompt


# --- ExecuteToolContext.replace() ---


@pytest.fixture
def tool_context(agent):
    return ExecuteToolContext(
        agent=agent,
        tool=None,
        tool_use={"toolUseId": "t1", "name": "calc", "input": {"x": 1}},
        invocation_state={},
        cancel_signal=threading.Event(),
        _interrupt_state=_InterruptState(),
    )


def test_tool_replace_changes_tool_use(tool_context):
    modified = tool_context.replace(tool_use={"toolUseId": "t1", "name": "calc", "input": {"x": 2}})

    assert modified.tool_use["input"] == {"x": 2}
    assert tool_context.tool_use["input"] == {"x": 1}


def test_tool_replace_preserves_interrupt_state(tool_context):
    """replace() must carry the internal interrupt state forward so interrupt() still resolves."""
    modified = tool_context.replace(tool_use={"toolUseId": "t1", "name": "calc", "input": {}})

    assert modified._interrupt_state is tool_context._interrupt_state
