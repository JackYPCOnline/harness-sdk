from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import ANY

from strands.background_tasks.in_process._engine import InProcessTaskEngine
from strands.background_tasks.in_process._types import (
    InProcessTaskExecutionContext,
    InProcessTaskExecutionOutcome,
    InProcessTaskRecord,
)
from strands.types.tools import ToolResult


def create_engine(
    execute: Callable[[InProcessTaskExecutionContext[str]], Awaitable[InProcessTaskExecutionOutcome[ToolResult]]],
    *,
    max_concurrency: int = 2,
    timeout: float = 1.0,
    on_task_updated: Callable[[InProcessTaskRecord[str, ToolResult]], None] | None = None,
) -> InProcessTaskEngine[str, ToolResult]:
    return InProcessTaskEngine(
        max_concurrency=max_concurrency,
        timeout=timeout,
        execute=execute,
        on_task_updated=on_task_updated or (lambda _: None),
    )


def create_admission(value: str) -> str:
    return value


def create_result(value: str) -> ToolResult:
    return {
        "toolUseId": value,
        "status": "success",
        "content": [{"text": value}],
    }


def create_state(value: str) -> dict[str, Any]:
    return {
        "interrupts": {
            value: {
                "id": value,
                "name": value,
            }
        },
        "activated": True,
    }


def get_state_value(state: dict[str, Any]) -> str:
    return next(iter(state["interrupts"]))


def assert_task(
    actual: InProcessTaskRecord[str, ToolResult] | None,
    task: InProcessTaskRecord[str, ToolResult],
    fields: dict[str, Any],
) -> None:
    exp_task = {
        "task_id": task["task_id"],
        "input": task["input"],
        **fields,
        "created_at": task["created_at"],
        "last_updated_at": ANY,
    }
    assert actual == exp_task


def deferred() -> tuple[
    asyncio.Future[InProcessTaskExecutionOutcome[ToolResult]],
    Callable[[InProcessTaskExecutionOutcome[ToolResult]], None],
]:
    future = asyncio.get_running_loop().create_future()
    return future, future.set_result
