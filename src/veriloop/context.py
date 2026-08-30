"""Deterministic, provider-independent projection of canonical agent history."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from typing import Any, Sequence

from .protocol import Message, Role, ToolCall


DEFAULT_CONTEXT_CHARS_SOFT_LIMIT = 60_000
DEFAULT_RECENT_GROUP_COUNT = 4
CONTEXT_TRUNCATION_MARKER = "...[veriloop context truncated]..."


@dataclass(frozen=True, slots=True)
class _HistoryGroup:
    messages: tuple[Message, ...]
    contains_verification_failure: bool


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    """Prune only complete history groups while retaining host-owned anchors."""

    context_chars_soft_limit: int = DEFAULT_CONTEXT_CHARS_SOFT_LIMIT
    recent_group_count: int = DEFAULT_RECENT_GROUP_COUNT

    def __post_init__(self) -> None:
        if (
            not isinstance(self.context_chars_soft_limit, int)
            or isinstance(self.context_chars_soft_limit, bool)
            or self.context_chars_soft_limit < len(CONTEXT_TRUNCATION_MARKER)
        ):
            raise ValueError(
                "context_chars_soft_limit must be an integer large enough for "
                "the truncation marker"
            )
        if (
            not isinstance(self.recent_group_count, int)
            or isinstance(self.recent_group_count, bool)
            or self.recent_group_count < 1
        ):
            raise ValueError("recent_group_count must be a positive integer")

    def estimate_chars(self, messages: Sequence[Message]) -> int:
        return sum(_message_chars(message) for message in messages)

    def project(self, history: Sequence[Message]) -> list[Message]:
        source = tuple(history)
        anchors, groups = _partition_history(source)
        if self.estimate_chars(source) <= self.context_chars_soft_limit:
            return deepcopy(list(source))

        protected_group_indexes = set(
            range(
                max(0, len(groups) - self.recent_group_count),
                len(groups),
            )
        )
        for index in range(len(groups) - 1, -1, -1):
            if groups[index].contains_verification_failure:
                protected_group_indexes.add(index)
                break

        kept = [True] * len(groups)
        for index in range(len(groups)):
            if index in protected_group_indexes:
                continue
            if self._selection_chars(anchors, groups, kept) <= (
                self.context_chars_soft_limit
            ):
                break
            kept[index] = False

        selected = [*anchors]
        for keep, group in zip(kept, groups, strict=True):
            if keep:
                selected.extend(group.messages)
        projected = deepcopy(selected)
        return _truncate_projection(projected, self.context_chars_soft_limit)

    def _selection_chars(
        self,
        anchors: tuple[Message, Message],
        groups: tuple[_HistoryGroup, ...],
        kept: list[bool],
    ) -> int:
        total = self.estimate_chars(anchors)
        for keep, group in zip(kept, groups, strict=True):
            if keep:
                total += self.estimate_chars(group.messages)
        return total


def _partition_history(
    history: tuple[Message, ...],
) -> tuple[tuple[Message, Message], tuple[_HistoryGroup, ...]]:
    if len(history) < 2:
        raise ValueError("history must contain initial system and user messages")
    if history[0].role is not Role.SYSTEM or history[1].role is not Role.USER:
        raise ValueError("history must begin with system and user messages")

    groups: list[_HistoryGroup] = []
    index = 2
    while index < len(history):
        message = history[index]
        if message.role is Role.TOOL:
            raise ValueError("history contains an orphan tool message")
        if message.role is not Role.ASSISTANT:
            raise ValueError("history after the initial task must contain assistant groups")
        if not message.tool_calls:
            groups.append(
                _HistoryGroup(
                    messages=(message,),
                    contains_verification_failure=False,
                )
            )
            index += 1
            continue

        call_ids = tuple(call.id for call in message.tool_calls)
        if len(set(call_ids)) != len(call_ids):
            raise ValueError("assistant tool call ids must be unique within a group")
        tool_messages: list[Message] = []
        cursor = index + 1
        while cursor < len(history) and history[cursor].role is Role.TOOL:
            tool_messages.append(history[cursor])
            cursor += 1
        result_ids = tuple(
            tool_message.tool_result.call_id
            for tool_message in tool_messages
            if tool_message.tool_result is not None
        )
        if (
            len(result_ids) != len(tool_messages)
            or len(result_ids) != len(call_ids)
            or result_ids != call_ids
        ):
            raise ValueError(
                "assistant tool calls and following tool results must form a "
                "complete group"
            )
        group_messages = (message, *tool_messages)
        groups.append(
            _HistoryGroup(
                messages=group_messages,
                contains_verification_failure=any(
                    _is_verification_failure(item) for item in tool_messages
                ),
            )
        )
        index = cursor
    return (history[0], history[1]), tuple(groups)


def _is_verification_failure(message: Message) -> bool:
    result = message.tool_result
    return bool(
        result is not None
        and not result.ok
        and result.metadata.get("verification") is True
        and result.metadata.get("verified") is False
    )


def _message_chars(message: Message) -> int:
    total = len(message.content)
    for call in message.tool_calls:
        total += len(call.id) + len(call.name) + _arguments_chars(call.arguments)
    if message.tool_result is not None:
        total += len(message.tool_result.call_id) + len(message.tool_result.tool_name)
    return total


def _arguments_chars(arguments: dict[str, Any]) -> int:
    return len(
        json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _truncate_projection(messages: list[Message], limit: int) -> list[Message]:
    while sum(_message_chars(message) for message in messages) > limit:
        overflow = sum(_message_chars(message) for message in messages) - limit
        candidates: list[tuple[int, int, int, int | None]] = []
        order = 0
        for message_index, message in enumerate(messages):
            content_reduction = len(message.content) - len(CONTEXT_TRUNCATION_MARKER)
            if content_reduction > 0:
                candidates.append(
                    (content_reduction, -order, message_index, None)
                )
            order += 1
            for call_index, call in enumerate(message.tool_calls):
                arguments_size = _arguments_chars(call.arguments)
                minimum_size = _arguments_chars(
                    {"_context_truncated": CONTEXT_TRUNCATION_MARKER}
                )
                if arguments_size > minimum_size:
                    candidates.append(
                        (
                            arguments_size - minimum_size,
                            -order,
                            message_index,
                            call_index,
                        )
                    )
                order += 1
        if not candidates:
            # The budget is soft: role structure, anchors, call ids, and complete
            # assistant/tool groups take precedence over an impossible hard cap.
            break

        reduction, _, message_index, call_index = max(candidates)
        requested_reduction = min(overflow, reduction)
        message = messages[message_index]
        if call_index is None:
            target = max(
                len(CONTEXT_TRUNCATION_MARKER),
                len(message.content) - requested_reduction,
            )
            content = _truncate_text(message.content, target)
            tool_result = message.tool_result
            if tool_result is not None:
                tool_result = replace(tool_result, content=content)
            messages[message_index] = replace(
                message,
                content=content,
                tool_result=tool_result,
            )
            continue

        calls = list(message.tool_calls)
        call = calls[call_index]
        target = max(
            _arguments_chars({"_context_truncated": CONTEXT_TRUNCATION_MARKER}),
            _arguments_chars(call.arguments) - requested_reduction,
        )
        calls[call_index] = replace(
            call,
            arguments=_truncate_arguments(call.arguments, target),
        )
        messages[message_index] = replace(message, tool_calls=tuple(calls))
    return messages


def _truncate_text(text: str, target_chars: int) -> str:
    if len(text) <= target_chars:
        return text
    available = target_chars - len(CONTEXT_TRUNCATION_MARKER)
    if available <= 0:
        return CONTEXT_TRUNCATION_MARKER
    head = (available + 1) // 2
    tail = available - head
    suffix = text[-tail:] if tail else ""
    return text[:head] + CONTEXT_TRUNCATION_MARKER + suffix


def _truncate_arguments(
    arguments: dict[str, Any],
    target_chars: int,
) -> dict[str, Any]:
    projected = deepcopy(arguments)
    while _arguments_chars(projected) > target_chars:
        leaves = _string_leaves(projected)
        reducible = [
            (len(value) - len(CONTEXT_TRUNCATION_MARKER), path, value)
            for path, value in leaves
            if len(value) > len(CONTEXT_TRUNCATION_MARKER)
        ]
        if not reducible:
            return {"_context_truncated": CONTEXT_TRUNCATION_MARKER}
        reduction, path, value = max(
            reducible,
            key=lambda item: (item[0], tuple(str(part) for part in item[1])),
        )
        overflow = _arguments_chars(projected) - target_chars
        target = max(
            len(CONTEXT_TRUNCATION_MARKER),
            len(value) - min(overflow, reduction),
        )
        _set_path(projected, path, _truncate_text(value, target))
    return projected


def _string_leaves(
    value: Any,
    path: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        leaves: list[tuple[tuple[str | int, ...], str]] = []
        for key in sorted(value):
            leaves.extend(_string_leaves(value[key], (*path, key)))
        return leaves
    if isinstance(value, list):
        leaves = []
        for index, item in enumerate(value):
            leaves.extend(_string_leaves(item, (*path, index)))
        return leaves
    return []


def _set_path(root: dict[str, Any], path: tuple[str | int, ...], value: str) -> None:
    target: Any = root
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
