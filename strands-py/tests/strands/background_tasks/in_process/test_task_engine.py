from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

from strands.background_tasks.in_process._task_engine import TaskEngine, TaskNotFoundError
from strands.background_tasks.in_process._task_types import TaskExecutionContext, TaskOutcome


@dataclass(frozen=True)
class _Input:
    value: str


@pytest.mark.asyncio
async def test_generic_engine_preserves_consumer_owned_input_and_result() -> None:
    input = _Input("work")
    result = threading.Lock()

    async def execute(context: TaskExecutionContext[_Input]) -> TaskOutcome[object]:
        assert context.input is input
        return {"status": "completed", "result": result}

    engine = TaskEngine[_Input, object](
        max_concurrency=1,
        timeout=1,
        execute=execute,
        on_task_updated=lambda _: None,
    )
    admitted = engine.submit(input)
    assert admitted["input"] is input

    await engine.wait_for_idle()

    completed = engine.get(admitted["task_id"])
    assert completed is not None
    assert completed["result"] is result
    completed["status"] = "cancelled"
    stored = engine.get(admitted["task_id"])
    assert stored is not None
    assert stored["status"] == "completed"


def test_generic_engine_reports_missing_tasks_without_background_tool_types() -> None:
    async def execute(context: TaskExecutionContext[str]) -> TaskOutcome[str]:
        return {"status": "completed", "result": context.input}

    engine = TaskEngine[str, str](
        max_concurrency=1,
        timeout=1,
        execute=execute,
        on_task_updated=lambda _: None,
    )

    with pytest.raises(TaskNotFoundError, match="was not found"):
        engine.cancel("missing", reason="stop")
