"""One-shot, storage-independent dreaming consolidation engine."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Mapping, Sequence

from .llm import LLMClient
from .models import DreamedMemory, MemoryBatch, MemoryType, SchemaError


@dataclass(frozen=True, slots=True)
class DreamingConfig:
    """Controls one second-stage extraction call."""

    max_input_tokens: int = 30_000
    max_output_items: int = 10
    retries: int = 3
    important_memory_definition: str = (
        "用户身份、关键决策、长期承诺，以及不可替代且长期有效的事实性内容"
    )

    def __post_init__(self) -> None:
        if self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        if self.max_output_items <= 0:
            raise ValueError("max_output_items must be positive")
        if self.retries <= 0:
            raise ValueError("retries must be positive")


@dataclass(frozen=True, slots=True)
class DreamingResult:
    """Second-stage memories and the exact input coverage used to produce them."""

    memories: tuple[DreamedMemory, ...]
    input_memory_ids: tuple[str, ...]
    omitted_memory_ids: tuple[str, ...] = ()
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "memories": [memory.to_dict() for memory in self.memories],
            "input_memory_ids": list(self.input_memory_ids),
            "omitted_memory_ids": list(self.omitted_memory_ids),
        }


def _estimate_tokens(text: str) -> int:
    """Keep the original dreaming approximation: roughly four chars per token."""

    return max(1, len(text) // 4)


def _memory_line(memory: Any) -> str:
    data = {
        "memory_id": memory.memory_id,
        "mem_type": memory.mem_type.value,
        "content": memory.content,
        "is_important": memory.is_important,
    }
    if memory.source_session_id:
        data["source_session_id"] = memory.source_session_id
    if memory.created_at:
        data["created_at"] = memory.created_at
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _fit_to_budget(memories: Sequence[Any], max_tokens: int) -> tuple[list[Any], list[str]]:
    """Mirror Agent Memory's middle-drop compression while reporting omissions."""

    selected = list(memories)
    removed_ids: set[str] = set()
    while len(selected) > 2 and _estimate_tokens("\n".join(_memory_line(item) for item in selected)) > max_tokens:
        removed = selected.pop(len(selected) // 2)
        removed_ids.add(removed.memory_id)
    estimated = _estimate_tokens("\n".join(_memory_line(item) for item in selected))
    if estimated > max_tokens:
        raise SchemaError(
            "the first and last retained memories exceed max_input_tokens "
            f"({estimated} > {max_tokens}); increase the limit or shorten the input"
        )
    omitted = [item.memory_id for item in memories if item.memory_id in removed_ids]
    return selected, omitted


def _parse_json_text(text: str) -> Any:
    if not isinstance(text, str) or not text.strip():
        raise SchemaError("LLM returned empty output")
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    candidate = match.group(1).strip() if match else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"LLM output is not valid JSON: {exc.msg}") from exc


def _as_important(value: Any) -> bool:
    """Retain the original dreaming module's conservative coercion."""

    return value is True or value == 1 or (isinstance(value, str) and value.strip().lower() == "true")


def _validate_output(parsed: Any, known_ids: Sequence[str], max_items: int) -> tuple[DreamedMemory, ...]:
    if isinstance(parsed, Mapping) and "memories" in parsed:
        parsed = parsed["memories"]
    if not isinstance(parsed, list):
        raise SchemaError("LLM output must be a JSON array")

    known_id_set = set(known_ids)
    result: list[DreamedMemory] = []
    positions = {memory_id: index for index, memory_id in enumerate(known_ids)}
    for index, item in enumerate(parsed[:max_items]):
        if not isinstance(item, Mapping):
            raise SchemaError(f"output item {index} must be an object")
        try:
            mem_type = MemoryType(item.get("mem_type"))
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"output item {index} has invalid mem_type") from exc
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            raise SchemaError(f"output item {index} has empty content")
        source_ids = item.get("source_memory_ids")
        if not isinstance(source_ids, list) or not source_ids:
            raise SchemaError(f"output item {index} must have non-empty source_memory_ids")
        if any(not isinstance(value, str) or value not in known_id_set for value in source_ids):
            raise SchemaError(f"output item {index} references an unknown source_memory_id")
        unique_sources = tuple(dict.fromkeys(source_ids))
        result.append(
            DreamedMemory(
                mem_type=mem_type,
                content=content.strip(),
                source_memory_ids=unique_sources,
                is_important=_as_important(item.get("is_important", False)),
            )
        )

    # Exact duplicates are merged deterministically, retaining all provenance.
    merged: dict[tuple[MemoryType, str], DreamedMemory] = {}
    for item in result:
        key = (item.mem_type, item.content)
        previous = merged.get(key)
        if previous is None:
            merged[key] = item
            continue
        sources = tuple(
            sorted(set(previous.source_memory_ids + item.source_memory_ids), key=lambda value: positions.get(value, 0))
        )
        merged[key] = DreamedMemory(
            mem_type=item.mem_type,
            content=item.content,
            source_memory_ids=sources,
            is_important=previous.is_important or item.is_important,
        )
    return tuple(merged.values())


class DreamingExtractor:
    """Consolidate a batch of extracted memories without storage or scheduling."""

    def __init__(self, llm: LLMClient, config: DreamingConfig | None = None) -> None:
        if not isinstance(llm, LLMClient):
            raise TypeError("llm must implement async complete(prompt) -> str")
        self._llm = llm
        self._config = config or DreamingConfig()
        self._template = files("agent_dreaming.prompts").joinpath("memory_consolidation.md").read_text(encoding="utf-8")

    def build_prompt(self, batch: MemoryBatch) -> tuple[str, tuple[str, ...]]:
        selected, omitted = _fit_to_budget(batch.memories, self._config.max_input_tokens)
        body = "\n".join(_memory_line(memory) for memory in selected)
        prompt = self._template
        variables = {
            "max_items": str(self._config.max_output_items),
            "important_memory_definition": self._config.important_memory_definition,
            "memories": body,
        }
        for name, value in variables.items():
            prompt = prompt.replace("{{" + name + "}}", value)
        return prompt, tuple(omitted)

    async def dream(self, batch: MemoryBatch | Mapping[str, Any]) -> DreamingResult:
        """Run second-stage extraction and return only validated, traceable output."""

        if not isinstance(batch, MemoryBatch):
            batch = MemoryBatch.from_dict(batch)
        if not batch.memories:
            return DreamingResult(memories=(), input_memory_ids=())

        prompt, omitted = self.build_prompt(batch)
        omitted_set = set(omitted)
        selected_ids = tuple(item.memory_id for item in batch.memories if item.memory_id not in omitted_set)
        last_error: Exception | None = None
        attempt_prompt = prompt
        for attempt in range(self._config.retries):
            try:
                response = await self._llm.complete(attempt_prompt)
                memories = _validate_output(_parse_json_text(response), selected_ids, self._config.max_output_items)
                return DreamingResult(
                    memories=memories,
                    input_memory_ids=selected_ids,
                    omitted_memory_ids=omitted,
                )
            except (SchemaError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self._config.retries:
                    attempt_prompt = (
                        prompt
                        + "\n\n上一次输出未通过格式校验："
                        + str(exc)
                        + "。请重新输出完整、合法且不含解释文字的 JSON 数组。"
                    )
        raise SchemaError(f"dreaming extraction failed after {self._config.retries} attempts: {last_error}")
