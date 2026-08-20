"""一次性执行、与存储无关的 Dreaming 记忆整合引擎。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from typing import Any, Mapping, Sequence

from .llm import LLMClient
from .models import Memory, MemoryBatch, MemoryType, SchemaError
from .preprocessing import preprocess_memory_batch


@dataclass(frozen=True, slots=True)
class DreamingConfig:
    """控制单次二阶段萃取调用。"""

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
        if not isinstance(self.important_memory_definition, str) or not self.important_memory_definition.strip():
            raise ValueError("important_memory_definition must be non-empty")


@dataclass(frozen=True, slots=True)
class DreamingResult:
    """二阶段记忆，以及生成这些记忆时实际覆盖的输入范围。"""

    memories: tuple[Memory, ...]
    input_memory_ids: tuple[str, ...]
    omitted_memory_ids: tuple[str, ...] = ()
    schema_version: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "memories": [memory.to_dict() for memory in self.memories],
        }

    def diagnostics_dict(self) -> dict[str, list[str]]:
        """将执行覆盖信息与可复用的记忆文档分开返回。"""

        return {
            "input_memory_ids": list(self.input_memory_ids),
            "omitted_memory_ids": list(self.omitted_memory_ids),
        }


def _estimate_tokens(text: str) -> int:
    """沿用原 Dreaming 估算方式：每个 token 约对应四个字符。"""

    return max(1, len(text) // 4)


def _memory_line(memory: Any) -> str:
    return json.dumps(memory.to_dict(), ensure_ascii=False, separators=(",", ":"))


def _fit_to_budget(memories: Sequence[Any], max_tokens: int) -> tuple[list[Any], list[str]]:
    """复用 Agent Memory 的中段丢弃压缩策略，并报告被省略项。"""

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


def _stable_memory_id(mem_type: MemoryType, content: str) -> str:
    seed = f"{mem_type.value}\0{content}".encode("utf-8")
    return "dream-" + hashlib.sha256(seed).hexdigest()


def _validate_output(
    parsed: Any,
    inputs: Sequence[Memory],
    max_items: int,
) -> tuple[Memory, ...]:
    if not isinstance(parsed, list):
        raise SchemaError("LLM output must be a JSON array")

    input_by_id = {memory.memory_id: memory for memory in inputs}
    positions = {memory.memory_id: index for index, memory in enumerate(inputs)}
    candidates: list[tuple[MemoryType, str, tuple[str, ...], bool]] = []
    for index, item in enumerate(parsed[:max_items]):
        if not isinstance(item, Mapping):
            raise SchemaError(f"output item {index} must be an object")
        unknown = sorted(set(item) - {"mem_type", "content", "source_memory_ids", "is_important"})
        if unknown:
            raise SchemaError(f"output item {index} has unknown fields: {', '.join(unknown)}")
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
        if any(not isinstance(value, str) or value not in input_by_id for value in source_ids):
            raise SchemaError(f"output item {index} references an unknown source_memory_id")
        important = item.get("is_important")
        if not isinstance(important, bool):
            raise SchemaError(f"output item {index} has non-boolean is_important")
        unique_sources = tuple(dict.fromkeys(source_ids))
        candidates.append((mem_type, content.strip(), unique_sources, important))

    # 对完全重复项做确定性合并，同时保留全部来源信息。
    merged: dict[tuple[MemoryType, str], tuple[list[str], bool]] = {}
    for mem_type, content, source_ids, important in candidates:
        key = (mem_type, content)
        previous = merged.get(key)
        if previous is None:
            merged[key] = (list(source_ids), important)
            continue
        direct_sources = sorted(
            set(previous[0] + list(source_ids)), key=lambda value: positions[value]
        )
        merged[key] = (direct_sources, previous[1] or important)

    result: list[Memory] = []
    for (mem_type, content), (direct_sources, important) in merged.items():
        lineage: list[str] = []
        session_ids: set[str] = set()
        timestamps: list[str] = []
        topics: set[str] = set()
        subtopics: set[str] = set()
        for source_id in direct_sources:
            source = input_by_id[source_id]
            lineage.extend(source.source_memory_ids or (source.memory_id,))
            session_ids.update(source.source_session_ids)
            if source.created_at is not None:
                timestamps.append(source.created_at)
            if source.topic is not None:
                topics.add(source.topic)
            if source.subtopic is not None:
                subtopics.add(source.subtopic)
        latest_timestamp = max(timestamps, key=datetime.fromisoformat) if timestamps else None
        inherited_topic = next(iter(topics)) if len(topics) == 1 else None
        inherited_subtopic = (
            next(iter(subtopics))
            if inherited_topic is not None and len(subtopics) == 1
            else None
        )
        result.append(
            Memory(
                memory_id=_stable_memory_id(mem_type, content),
                mem_type=mem_type,
                content=content,
                source_memory_ids=tuple(dict.fromkeys(lineage)),
                is_important=important,
                source_session_ids=tuple(sorted(session_ids)),
                created_at=latest_timestamp,
                topic=inherited_topic,
                subtopic=inherited_subtopic,
            )
        )
    return tuple(result)


class DreamingExtractor:
    """在不依赖存储和调度的情况下，整合一批已提取记忆。"""

    def __init__(self, llm: LLMClient, config: DreamingConfig | None = None) -> None:
        if not isinstance(llm, LLMClient):
            raise TypeError("llm must implement async complete(prompt) -> str")
        self._llm = llm
        self._config = config or DreamingConfig()
        self._template = files("agent_dreaming.prompts").joinpath("memory_consolidation.md").read_text(encoding="utf-8")

    def _build_prompt_for_memories(
        self, memories: Sequence[Memory]
    ) -> tuple[str, tuple[str, ...]]:
        selected, omitted = _fit_to_budget(memories, self._config.max_input_tokens)
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

    def build_prompt(self, batch: MemoryBatch) -> tuple[str, tuple[str, ...]]:
        preprocessed = preprocess_memory_batch(batch)
        prompt, budget_omitted = self._build_prompt_for_memories(preprocessed.batch.memories)
        omitted_set = set(preprocessed.removed_memory_ids) | set(budget_omitted)
        omitted = tuple(
            memory.memory_id for memory in batch.memories if memory.memory_id in omitted_set
        )
        return prompt, omitted

    async def dream(self, batch: MemoryBatch | Mapping[str, Any]) -> DreamingResult:
        """执行二阶段萃取，仅返回通过校验且可追溯的输出。"""

        if not isinstance(batch, MemoryBatch):
            batch = MemoryBatch.from_dict(batch)
        if not batch.memories:
            return DreamingResult(memories=(), input_memory_ids=())

        preprocessed = preprocess_memory_batch(batch)
        prompt, budget_omitted = self._build_prompt_for_memories(preprocessed.batch.memories)
        omitted_set = set(preprocessed.removed_memory_ids) | set(budget_omitted)
        omitted = tuple(
            memory.memory_id for memory in batch.memories if memory.memory_id in omitted_set
        )
        selected_ids = tuple(
            item.memory_id
            for item in preprocessed.batch.memories
            if item.memory_id not in set(budget_omitted)
        )
        selected_id_set = set(selected_ids)
        selected = tuple(
            item for item in preprocessed.batch.memories if item.memory_id in selected_id_set
        )
        last_error: Exception | None = None
        attempt_prompt = prompt
        for attempt in range(self._config.retries):
            try:
                response = await self._llm.complete(attempt_prompt)
                memories = _validate_output(_parse_json_text(response), selected, self._config.max_output_items)
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
