"""记忆进入 Dreaming 引擎前，按类型执行独立预处理。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

from .models import Memory, MemoryBatch, MemoryType


CORE_TYPES = frozenset({"core", "user_profile"})
EPISODIC_TYPES = frozenset({"episodic", "episodic_memory"})


def content_sha256(content: str) -> str:
    """返回用于 Core 确定性去重的精确内容摘要。"""

    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


def _parse_timestamp(value: str, *, field: str, row_number: int) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"memory CSV row {row_number} has invalid {field}: {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _recency_key(
    row: Mapping[str, str], row_index: int, row_number: int
) -> tuple[datetime, datetime, int]:
    updated_at = _parse_timestamp(
        row.get("updated_at", ""), field="updated_at", row_number=row_number
    )
    created_at = _parse_timestamp(
        row.get("created_at", ""), field="created_at", row_number=row_number
    )
    minimum = datetime.min.replace(tzinfo=timezone.utc)
    effective_time = updated_at or created_at or minimum
    return effective_time, created_at or minimum, row_index


def _selected_core_indices(
    rows: Sequence[Mapping[str, str]], row_numbers: Sequence[int] | None = None
) -> set[int]:
    selected: dict[str, tuple[tuple[datetime, datetime, int], int]] = {}
    for index, row in enumerate(rows):
        digest = content_sha256(row.get("content", ""))
        row_number = row_numbers[index] if row_numbers is not None else index + 2
        candidate = (_recency_key(row, index, row_number), index)
        previous = selected.get(digest)
        if previous is None or candidate[0] > previous[0]:
            selected[digest] = candidate
    return {index for _, index in selected.values()}


def preprocess_core_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """按内容精确哈希去重 Core 行，并保留最新的一行。"""

    selected = _selected_core_indices(rows)
    return [dict(row) for index, row in enumerate(rows) if index in selected]


def preprocess_episodic_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """在定义 Episodic 专属规则前，原样保留其数据行。"""

    return [dict(row) for row in rows]


def preprocess_memory_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """按记忆类型分派预处理，同时保持稳定的行顺序。"""

    core_positions = [
        index
        for index, row in enumerate(rows)
        if row.get("type", "").strip().casefold() in CORE_TYPES
    ]
    core_rows = [rows[index] for index in core_positions]
    selected_core_offsets = _selected_core_indices(
        core_rows, [index + 2 for index in core_positions]
    )
    selected_core_positions = {
        core_positions[offset] for offset in selected_core_offsets
    }

    result: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        source_type = row.get("type", "").strip().casefold()
        if source_type in CORE_TYPES and index not in selected_core_positions:
            continue
        result.append(dict(row))
    return result


@dataclass(frozen=True, slots=True)
class MemoryPreprocessingResult:
    """规范记忆的预处理结果，以及生成提示词前移除的记忆 ID。"""

    batch: MemoryBatch
    removed_memory_ids: tuple[str, ...] = ()


def _memory_recency_key(memory: Memory, index: int) -> tuple[datetime, int]:
    timestamp = (
        datetime.fromisoformat(memory.created_at)
        if memory.created_at is not None
        else datetime.min.replace(tzinfo=timezone.utc)
    )
    return timestamp, index


def preprocess_core_memories(memories: Sequence[Memory]) -> MemoryPreprocessingResult:
    """按哈希去重规范 Core 记忆，并保留最新记录。"""

    selected: dict[str, tuple[tuple[datetime, int], int]] = {}
    for index, memory in enumerate(memories):
        digest = content_sha256(memory.content)
        candidate = (_memory_recency_key(memory, index), index)
        previous = selected.get(digest)
        if previous is None or candidate[0] > previous[0]:
            selected[digest] = candidate
    selected_indices = {index for _, index in selected.values()}
    kept = tuple(memory for index, memory in enumerate(memories) if index in selected_indices)
    removed = tuple(
        memory.memory_id for index, memory in enumerate(memories) if index not in selected_indices
    )
    return MemoryPreprocessingResult(MemoryBatch(kept), removed)


def preprocess_episodic_memories(memories: Sequence[Memory]) -> MemoryPreprocessingResult:
    """在定义 Episodic 专属规则前，原样保留规范记忆。"""

    return MemoryPreprocessingResult(MemoryBatch(tuple(memories)))


def preprocess_memory_batch(batch: MemoryBatch) -> MemoryPreprocessingResult:
    """按原批次顺序，对规范记忆执行分类型预处理。"""

    core_positions = [
        index
        for index, memory in enumerate(batch.memories)
        if memory.mem_type is MemoryType.USER_PROFILE
    ]
    core_result = preprocess_core_memories([batch.memories[index] for index in core_positions])
    kept_core_ids = {memory.memory_id for memory in core_result.batch.memories}

    episodic_positions = [
        index
        for index, memory in enumerate(batch.memories)
        if memory.mem_type is MemoryType.EPISODIC_MEMORY
    ]
    episodic_result = preprocess_episodic_memories(
        [batch.memories[index] for index in episodic_positions]
    )
    kept_episodic_ids = {memory.memory_id for memory in episodic_result.batch.memories}

    kept: list[Memory] = []
    removed: list[str] = []
    for memory in batch.memories:
        if memory.mem_type is MemoryType.USER_PROFILE and memory.memory_id not in kept_core_ids:
            removed.append(memory.memory_id)
            continue
        if (
            memory.mem_type is MemoryType.EPISODIC_MEMORY
            and memory.memory_id not in kept_episodic_ids
        ):
            removed.append(memory.memory_id)
            continue
        kept.append(memory)
    return MemoryPreprocessingResult(MemoryBatch(tuple(kept)), tuple(removed))
