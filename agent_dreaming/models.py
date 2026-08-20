"""Canonical public memory format for dreaming input and output."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Sequence


class SchemaError(ValueError):
    """Raised when caller input or model output violates the public schema."""


class MemoryType(str, Enum):
    """Memory types supported by the dreaming pipeline."""

    USER_PROFILE = "user_profile"
    SEMANTIC_MEMORY = "semantic_memory"
    EPISODIC_MEMORY = "episodic_memory"


def _reject_unknown_fields(data: Mapping[str, Any], allowed: set[str], subject: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise SchemaError(f"{subject} has unknown fields: {', '.join(unknown)}")


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{field_name} must be a non-empty string")
    return value.strip()


def _memory_type(value: Any, field_name: str = "mem_type") -> MemoryType:
    try:
        return MemoryType(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in MemoryType)
        raise SchemaError(f"{field_name} must be one of: {allowed}") from exc


def _memory_ids(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SchemaError(f"{field_name} must be an array")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_required_text(item, f"{field_name}[{index}]"))
    return tuple(dict.fromkeys(result))


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _optional_timestamp(value: Any) -> str | None:
    value = _optional_text(value, "created_at")
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SchemaError("created_at must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SchemaError("created_at must include a timezone offset")
    return value


@dataclass(frozen=True, slots=True)
class Memory:
    """One canonical memory record, used unchanged before and after dreaming."""

    memory_id: str
    mem_type: MemoryType
    content: str
    source_memory_ids: tuple[str, ...]
    is_important: bool
    source_session_ids: tuple[str, ...] = ()
    created_at: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Memory":
        if not isinstance(data, Mapping):
            raise SchemaError("each memories item must be an object")
        _reject_unknown_fields(
            data,
            {
                "memory_id",
                "mem_type",
                "content",
                "source_memory_ids",
                "is_important",
                "source_session_ids",
                "created_at",
            },
            "memory",
        )
        important = data.get("is_important")
        if not isinstance(important, bool):
            raise SchemaError("is_important must be a boolean")
        memory_id = _required_text(data.get("memory_id"), "memory_id")
        source_memory_ids = _memory_ids(data.get("source_memory_ids"), "source_memory_ids")
        if memory_id in source_memory_ids:
            raise SchemaError("source_memory_ids must not reference the memory itself")
        return cls(
            memory_id=memory_id,
            mem_type=_memory_type(data.get("mem_type")),
            content=_required_text(data.get("content"), "content"),
            source_memory_ids=source_memory_ids,
            is_important=important,
            source_session_ids=_memory_ids(data.get("source_session_ids", []), "source_session_ids"),
            created_at=_optional_timestamp(data.get("created_at")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "mem_type": self.mem_type.value,
            "content": self.content,
            "source_memory_ids": list(self.source_memory_ids),
            "is_important": self.is_important,
            "source_session_ids": list(self.source_session_ids),
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class MemoryBatch:
    """Versioned document containing canonical memory records."""

    memories: tuple[Memory, ...]
    schema_version: str = "2.0"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryBatch":
        if not isinstance(data, Mapping):
            raise SchemaError("input must be a JSON object")
        _reject_unknown_fields(data, {"schema_version", "memories"}, "input")
        version = data.get("schema_version")
        if version != "2.0":
            raise SchemaError(f"unsupported schema_version: {version!r}; expected '2.0'")
        raw_memories = data.get("memories")
        if not isinstance(raw_memories, Sequence) or isinstance(raw_memories, (str, bytes)):
            raise SchemaError("memories must be an array")
        memories = tuple(Memory.from_dict(item) for item in raw_memories)
        ids = [item.memory_id for item in memories]
        if len(ids) != len(set(ids)):
            raise SchemaError("memory_id values must be unique within a batch")
        return cls(schema_version=version, memories=memories)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "memories": [memory.to_dict() for memory in self.memories],
        }
