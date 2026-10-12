"""Generic facade for bounded in-process task execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Generic

from ._scheduler import TaskNotFoundError as TaskNotFoundError
from ._scheduler import _TaskScheduler
from ._task_types import (
    TaskCancelSignal,
    TaskExecutionContext,
    TaskOutcome,
    TaskRecord,
    TaskState,
    TInput,
    TResult,
)


class TaskEngine(Generic[TInput, TResult]):
    """Run and track bounded tasks with consumer-defined input and result types.

    The facade owns no scheduling behavior; it exposes the generic contract over the same
    private scheduler used by the compatibility-preserving tool-task engine.
    """

    def __init__(
        self,
        *,
        max_concurrency: int,
        timeout: float,
        execute: Callable[[TaskExecutionContext[TInput]], Awaitable[TaskOutcome[TResult]]],
        on_task_updated: Callable[[TaskRecord[TInput, TResult]], None],
    ) -> None:
        """Initialize the engine.

        Args:
            max_concurrency: Maximum concurrent physical executions.
            timeout: Per-execution timeout in seconds, or infinity.
            execute: Callback that executes one task.
            on_task_updated: Best-effort committed-update callback.

        Raises:
            TypeError: If execution configuration is invalid.
        """
        self._timeout = timeout
        self._scheduler = _TaskScheduler(
            max_concurrency=max_concurrency,
            timeout=timeout,
            execute=execute,
            on_task_updated=on_task_updated,
        )

    def submit(self, input: TInput) -> TaskRecord[TInput, TResult]:
        """Submit a task for execution."""
        return self._scheduler.submit(input)

    def get(self, task_id: str) -> TaskRecord[TInput, TResult] | None:
        """Return a task snapshot if it exists."""
        return self._scheduler.get(task_id)

    def list(self) -> list[TaskRecord[TInput, TResult]]:
        """Return all task snapshots."""
        return self._scheduler.list()

    def remove(self, task_id: str) -> None:
        """Remove a terminal task."""
        self._scheduler.remove(task_id)

    def cancel(self, task_id: str, *, reason: str) -> TaskRecord[TInput, TResult]:
        """Cancel a non-terminal task."""
        return self._scheduler.cancel(task_id, reason=reason)

    async def wait_for_idle(self, *, cancel_signal: TaskCancelSignal | None = None) -> None:
        """Wait until no task is queued or physically executing."""
        await self._scheduler.wait_for_idle(cancel_signal=cancel_signal)

    def resume(
        self,
        task_id: str,
        update: Callable[[TaskState], tuple[TaskState, bool]],
    ) -> TaskRecord[TInput, TResult]:
        """Apply input to a task that requires it."""
        return self._scheduler.resume(task_id, update)
