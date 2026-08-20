"""Dependency-free input and output models for the dreaming boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class SchemaError(ValueError):
    """Raised when caller input or model output violates the public schema."""


class MemoryType(str, Enum):
    """The fragment memory types supported by the original dreaming pipeline."""

    USER_PROFILE = "user_profile"
    SEMANTIC_MEMORY = "semantic_memory"
    EPISODIC_MEMORY = "episodic_memory"


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _memory_type(value: Any, field_name: str = "mem_type") -> MemoryType:
    try:
        return MemoryType(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in MemoryType)
        raise SchemaError(f"{field_name} must be one of: {allowed}") from exc


@dataclass(frozen=True, slots=True)
class ExtractedMemory:
    """One already-extracted memory supplied by the caller."""

    memory_id: str
    mem_type: MemoryType
    content: str
    source_session_id: str | None = None
    is_important: bool = False
    created_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExtractedMemory":
        if not isinstance(data, Mapping):
            raise SchemaError("each memories item must be an object")
        important = data.get("is_important", False)
        if not isinstance(important, bool):
            raise SchemaError("is_important must be a boolean")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise SchemaError("metadata must be an object")
        return cls(
            memory_id=_required_text(data.get("memory_id"), "memory_id"),
            mem_type=_memory_type(data.get("mem_type")),
            content=_required_text(data.get("content"), "content"),
            source_session_id=_optional_text(data.get("source_session_id"), "source_session_id"),
            is_important=important,
            created_at=_optional_text(data.get("created_at"), "created_at"),
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "memory_id": self.memory_id,
            "mem_type": self.mem_type.value,
            "content": self.content,
            "is_important": self.is_important,
        }
        if self.source_session_id is not None:
            result["source_session_id"] = self.source_session_id
        if self.created_at is not None:
            result["created_at"] = self.created_at
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


@dataclass(frozen=True, slots=True)
class MemoryBatch:
    """Versioned standalone input contract."""

    user_id: str
    memories: tuple[ExtractedMemory, ...]
    scope_id: str = "default"
    batch_id: str | None = None
    schema_version: str = "1.0"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryBatch":
        if not isinstance(data, Mapping):
            raise SchemaError("input must be a JSON object")
        version = data.get("schema_version", "1.0")
        if version != "1.0":
            raise SchemaError(f"unsupported schema_version: {version!r}")
        raw_memories = data.get("memories")
        if not isinstance(raw_memories, Sequence) or isinstance(raw_memories, (str, bytes)):
            raise SchemaError("memories must be an array")
        memories = tuple(ExtractedMemory.from_dict(item) for item in raw_memories)
        ids = [item.memory_id for item in memories]
        if len(ids) != len(set(ids)):
            raise SchemaError("memory_id values must be unique within a batch")
        return cls(
            user_id=_required_text(data.get("user_id"), "user_id"),
            scope_id=_required_text(data.get("scope_id", "default"), "scope_id"),
            batch_id=_optional_text(data.get("batch_id"), "batch_id"),
            schema_version=version,
            memories=memories,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "user_id": self.user_id,
            "scope_id": self.scope_id,
            "memories": [memory.to_dict() for memory in self.memories],
        }
        if self.batch_id is not None:
            result["batch_id"] = self.batch_id
        return result


@dataclass(frozen=True, slots=True)
class DreamedMemory:
    """A consolidated memory with explicit provenance."""

    mem_type: MemoryType
    content: str
    source_memory_ids: tuple[str, ...]
    is_important: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mem_type": self.mem_type.value,
            "content": self.content,
            "source_memory_ids": list(self.source_memory_ids),
            "is_important": self.is_important,
        }

