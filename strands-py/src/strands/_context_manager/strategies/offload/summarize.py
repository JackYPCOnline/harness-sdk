"""Summarize strategy — replaces oversized content with LLM-generated summaries."""

from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ....agent.conversation_manager.compression.pin_message import is_pinned
from ....hooks import AfterInvocationEvent, HookOrder
from ....types.content import ContentBlock, Message, Messages
from ....types.tools import ToolResult, ToolResultContent
from ...methods.summarize import (
    SummarizeConfig,
    _flatten_messages_to_content,
    _format_summarized,
    _summarize_content,
    _tool_result_to_content_blocks,
)
from ...stash import Stash, _format_stash_refs
from ...types import ContextState
from .base import (
    BaseOffloadStrategy,
    OffloadConditions,
    OffloadTarget,
    _build_conditions,
    _collect_removable_with_pair,
    _repair_alternation,
    _splice_with_pairs,
)

if TYPE_CHECKING:
    from ....agent.agent import Agent
    from ....models.model import Model

logger = logging.getLogger(__name__)

# Bound concurrent summarizer calls and advance additional eligible blocks on later passes.
_BACKGROUND_BATCH_SIZE = 10


@dataclass
class _PendingSummary:
    """A tool result whose summary is being produced in the background."""

    # A different block object at commit time means another strategy replaced the original.
    original: ContentBlock
    tokens: int
    task: asyncio.Task[str | None]


class SummarizeStrategy(BaseOffloadStrategy):
    """Summarize strategy — replaces oversized content with LLM-generated summaries."""

    @property
    def name(self) -> str:
        """Strategy name."""
        return "offload:summarize"

    def __init__(
        self,
        target: OffloadTarget | None = None,
        config: SummarizeConfig | None = None,
        conditions: OffloadConditions | None = None,
    ) -> None:
        """Initialize the strategy.

        Raises:
            ValueError: If ``background`` is combined with a ``utilization`` condition.
        """
        super().__init__(target, conditions)
        self._config: SummarizeConfig = config or {}
        self._background = bool(self._config.get("background", False))
        if self._background and self._is_message_level:
            raise ValueError("Background summarization applies to per-block strategies; remove 'utilization'")
        self._pending: dict[Agent, dict[str, _PendingSummary]] = {}
        self._overflow_agents: set[Agent] = set()

    def init(self, agent: Agent, stash: Stash | None = None) -> None:
        """Register eager processing and flush background summaries before session persistence."""
        super().init(agent, stash)
        if not self._background:
            return

        async def _flush_on_after_invocation(event: AfterInvocationEvent) -> None:
            context = ContextState(messages=event.agent.messages, agent=event.agent, utilization=0, stash=stash)
            await self._flush(context)

        agent.add_hook(_flush_on_after_invocation, AfterInvocationEvent, order=HookOrder.SDK_FIRST)

    def when(
        self,
        *,
        threshold: int | None = None,
        utilization: float | None = None,
        preserve_recent: int | float = 0,
    ) -> SummarizeStrategy:
        """Return a new instance with the given conditions applied."""
        return SummarizeStrategy(
            self._target,
            self._config,
            _build_conditions(threshold=threshold, utilization=utilization, preserve_recent=preserve_recent),
        )

    async def apply(self, context: ContextState) -> bool:
        """Apply summarization. Returns False if no model is available."""
        if not self._background:
            if not self._resolve_model(context.agent):
                logger.warning("strategy=<%s> | no model available for summarization", self.name)
                return False
            return await super().apply(context)

        committed = self._commit_ready(context, allow_partial=context.overflow)
        if context.overflow:
            await self._cancel_pending(context.agent)

        if not self._resolve_model(context.agent):
            logger.warning("strategy=<%s> | no model available for summarization", self.name)
            return committed

        if context.overflow:
            self._overflow_agents.add(context.agent)
        try:
            acted = await super().apply(context)
        finally:
            self._overflow_agents.discard(context.agent)
        return committed or acted

    async def _apply_per_message(self, context: ContextState) -> bool:
        """Summarize oldest eligible messages into a single summary message."""
        model = self._resolve_model(context.agent)
        if not model:
            return False

        messages = context.messages
        if len(messages) <= 1:
            return False

        eligible = await self._get_eligible_messages(context)
        if not eligible:
            return False

        identity_map = {id(msg): index for index, msg in enumerate(messages)}
        safe_ids: set[int] = set()
        for message in eligible:
            index = identity_map.get(id(message))
            if index is None:
                continue
            for removable in _collect_removable_with_pair(messages, index):
                safe_ids.add(id(removable))
        safe = [msg for msg in messages if id(msg) in safe_ids]
        if not safe:
            return False

        content_blocks = _flatten_messages_to_content(safe)
        summary = await _summarize_content(content_blocks, model, self._config)
        if not summary:
            return False

        total_tokens = await model.count_tokens(safe)

        removed, lowest_index = _splice_with_pairs(messages, safe)
        if removed == 0:
            return False

        summary_message = Message(
            role="user",
            content=[ContentBlock(text=_format_summarized(f"{removed} messages", total_tokens, summary))],
        )
        insert_index = max(1, min(lowest_index, len(messages)))
        messages.insert(insert_index, summary_message)

        _repair_alternation(messages)
        logger.debug("summarized=<%s>, tokens=<%s> | batched summarization complete", removed, total_tokens)
        return True

    async def _replace_block(
        self,
        block: ContentBlock,
        tokens: int,
        message: Message,
        agent: Agent,
        stash_refs: list[str],
    ) -> ContentBlock | None:
        model = self._resolve_model(agent)
        if not model:
            return None

        refs = _format_stash_refs(stash_refs)

        if "toolResult" in block:
            return await self._replace_tool_result(block, tokens, model, refs, agent)

        if "text" not in block:
            summary = await _summarize_content([block], model, self._config)
            if summary:
                logger.debug(
                    "tracking_id=<%s>, tokens=<%s> | summarized media block", message.get("tracking_id"), tokens
                )
                marker = _format_summarized("media block", tokens, summary) + refs
                return ContentBlock(text=marker)
            logger.debug("tracking_id=<%s>, tokens=<%s> | offloaded media block", message.get("tracking_id"), tokens)
            return ContentBlock(text=f"[Offloaded: ~{tokens} tokens]{refs}")

        summary = await _summarize_content([ContentBlock(text=block["text"])], model, self._config)
        if not summary:
            return None

        logger.debug("tracking_id=<%s>, tokens=<%s> | summarized text block", message.get("tracking_id"), tokens)
        marker = _format_summarized("text block", tokens, summary) + refs
        return ContentBlock(text=marker)

    async def _replace_tool_result(
        self,
        block: ContentBlock,
        tokens: int,
        model: Model,
        refs: str,
        agent: Agent,
    ) -> ContentBlock | None:
        if self._background and agent not in self._overflow_agents:
            self._submit(agent, block, tokens, model)
            return None

        tool_result = block["toolResult"]
        content_blocks = _tool_result_to_content_blocks(tool_result["content"])
        summary = await _summarize_content(content_blocks, model, self._config)
        if not summary:
            return None

        logger.debug("tool_use_id=<%s>, tokens=<%s> | summarized tool result", tool_result["toolUseId"], tokens)
        return _summarized_tool_result(tool_result, tokens, summary, refs)

    def _submit(self, agent: Agent, block: ContentBlock, tokens: int, model: Model) -> None:
        """Start summarizing a tool result in the background, leaving the block in place."""
        pending = self._pending.setdefault(agent, {})
        tool_result = block["toolResult"]
        tool_use_id = tool_result["toolUseId"]
        if tool_use_id in pending or len(pending) >= _BACKGROUND_BATCH_SIZE:
            return

        content_blocks = _tool_result_to_content_blocks(copy.deepcopy(tool_result["content"]))
        task = asyncio.create_task(
            _summarize_content(content_blocks, model, self._config),
            name=f"strands-context-summary-{tool_use_id}",
        )
        pending[tool_use_id] = _PendingSummary(original=block, tokens=tokens, task=task)
        logger.debug("tool_use_id=<%s>, tokens=<%s> | background summarization submitted", tool_use_id, tokens)

    async def _flush(self, context: ContextState) -> None:
        """Wait for one background batch and commit its completed summaries."""
        pending = self._pending.get(context.agent)
        if not pending:
            return
        await asyncio.gather(*(item.task for item in pending.values()), return_exceptions=True)
        self._commit_ready(context)

    async def _cancel_pending(self, agent: Agent) -> None:
        """Cancel summaries that overflow recovery will recompute inline."""
        pending = self._pending.pop(agent, {})
        for item in pending.values():
            item.task.cancel()
        await asyncio.gather(*(item.task for item in pending.values()), return_exceptions=True)

    def _commit_ready(self, context: ContextState, *, allow_partial: bool = False) -> bool:
        """Swap a finished summary batch into history. Returns True if any block changed."""
        pending = self._pending.get(context.agent)
        if not pending or (not allow_partial and any(not item.task.done() for item in pending.values())):
            return False

        ready = {tool_use_id: item for tool_use_id, item in pending.items() if item.task.done()}
        acted = False
        for tool_use_id, item in ready.items():
            del pending[tool_use_id]
            if self._commit_one(context, tool_use_id, item):
                acted = True
        if not pending:
            self._pending.pop(context.agent, None)
        return acted

    def _commit_one(self, context: ContextState, tool_use_id: str, pending: _PendingSummary) -> bool:
        if pending.task.cancelled():
            return False
        try:
            summary = pending.task.result()
        except Exception:
            logger.warning("tool_use_id=<%s> | background summarization failed", tool_use_id, exc_info=True)
            return False
        if not summary:
            return False

        location = _locate_tool_result(context.messages, tool_use_id)
        if location is None:
            logger.debug("tool_use_id=<%s> | background summary discarded, block no longer in history", tool_use_id)
            return False
        message_index, block_index = location
        message = context.messages[message_index]
        block = message["content"][block_index]
        if block is not pending.original or is_pinned(context.messages, message_index):
            logger.debug("tool_use_id=<%s> | background summary discarded, block replaced or pinned", tool_use_id)
            return False

        stash_refs = context.stash.refs_for(block, message, block_index) if context.stash else []
        message["content"][block_index] = _summarized_tool_result(
            block["toolResult"], pending.tokens, summary, _format_stash_refs(stash_refs)
        )
        logger.debug("tool_use_id=<%s>, tokens=<%s> | summarized tool result", tool_use_id, pending.tokens)
        return True

    def _resolve_model(self, agent: Agent) -> Model | None:
        return self._config.get("model") or agent.aux_model


def _summarized_tool_result(tool_result: ToolResult, tokens: int, summary: str, refs: str) -> ContentBlock:
    """Build the tool result block that stands in for a summarized one."""
    marker = _format_summarized("tool result", tokens, summary) + refs
    summarized_content: list[ToolResultContent] = [{"text": marker}]
    return ContentBlock(
        toolResult=ToolResult(
            toolUseId=tool_result["toolUseId"],
            status=tool_result["status"],
            content=summarized_content,
        )
    )


def _locate_tool_result(messages: Messages, tool_use_id: str) -> tuple[int, int] | None:
    """Return the (message index, block index) of a tool result, or None if it is no longer in history."""
    locations = (
        (message_index, block_index)
        for message_index, message in enumerate(messages)
        for block_index, block in enumerate(message["content"])
        if "toolResult" in block and block["toolResult"]["toolUseId"] == tool_use_id
    )
    return next(locations, None)
