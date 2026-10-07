from collections.abc import AsyncGenerator

from typing_extensions import assert_type

from strands import Agent
from strands.middleware import (
    ExecuteToolContext,
    ExecuteToolStage,
    InvokeModelContext,
    InvokeModelStage,
    MiddlewareNext,
    MiddlewareResult,
    ModelStopReason,
    ToolResultEvent,
)
from strands.types._events import TypedEvent
from strands.types.content import Message
from strands.types.event_loop import StopReason


def inject_prompt(context: InvokeModelContext) -> InvokeModelContext:
    return context.replace(system_prompt="Be concise.")


async def inject_prompt_async(context: InvokeModelContext) -> InvokeModelContext:
    assert_type(context.replace(model=context.model), InvokeModelContext)
    return context


def log_stop_reason(result: MiddlewareResult[ModelStopReason]) -> MiddlewareResult[ModelStopReason]:
    assert_type(result.value, ModelStopReason)
    assert_type(result.value.stop_reason, StopReason)
    assert_type(result.value.message, Message)
    return result.replace(value=result.value)


async def rewrite_stop_reason(result: MiddlewareResult[ModelStopReason]) -> MiddlewareResult[ModelStopReason]:
    return result


async def passthrough(
    context: InvokeModelContext, next_fn: MiddlewareNext[InvokeModelContext, TypedEvent]
) -> AsyncGenerator[TypedEvent, None]:
    async for event in next_fn(context):
        yield event


async def approval_gate(
    context: ExecuteToolContext, next_fn: MiddlewareNext[ExecuteToolContext, TypedEvent]
) -> AsyncGenerator[TypedEvent, None]:
    approval = context.interrupt("approve_tool", reason=context.tool_use["name"])
    if approval.response != "approved":
        yield ToolResultEvent({"toolUseId": context.tool_use["toolUseId"], "status": "error", "content": []})
        return
    async for event in next_fn(context.replace(invocation_state=context.invocation_state)):
        yield event


def unwrap_result(result: MiddlewareResult[ModelStopReason]) -> ModelStopReason:
    return result.value


def register_middleware(agent: Agent) -> None:
    assert_type(agent.add_middleware(InvokeModelStage, passthrough), None)
    agent.add_middleware(InvokeModelStage.Wrap, passthrough)
    agent.add_middleware(InvokeModelStage.Input, inject_prompt)
    agent.add_middleware(InvokeModelStage.Input, inject_prompt_async)
    agent.add_middleware(InvokeModelStage.Output, log_stop_reason)
    agent.add_middleware(InvokeModelStage.Output, rewrite_stop_reason)
    agent.add_middleware(ExecuteToolStage, approval_gate)

    agent.add_middleware(ExecuteToolStage.Input, inject_prompt)  # type: ignore[arg-type]
    agent.add_middleware(ExecuteToolStage, passthrough)  # type: ignore[arg-type]
    agent.add_middleware(InvokeModelStage, inject_prompt)  # type: ignore[arg-type]
    agent.add_middleware(InvokeModelStage.Output, inject_prompt)  # type: ignore[arg-type]
    agent.add_middleware(InvokeModelStage.Output, unwrap_result)  # type: ignore[arg-type]
