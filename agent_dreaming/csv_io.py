"""独立基准测试所用紧凑记忆 CSV 的适配器。"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from .models import MemoryBatch, SchemaError
from .preprocessing import preprocess_memory_rows


MEMORY_CSV_COLUMNS = ("id", "type", "topic", "subtopic", "content")
TIMESTAMPED_MEMORY_CSV_COLUMNS = MEMORY_CSV_COLUMNS + ("created_at", "updated_at")
_TYPE_ALIASES = {
    "core": "user_profile",
    "user_profile": "user_profile",
    "semantic": "semantic_memory",
    "semantic_memory": "semantic_memory",
    "episodic": "episodic_memory",
    "episodic_memory": "episodic_memory",
}
_CSV_TYPES = {
    "user_profile": "core",
    "semantic_memory": "semantic",
    "episodic_memory": "episodic",
}


def memory_batch_from_csv(path: str | Path) -> MemoryBatch:
    """将紧凑 CSV 数据行加载为 Dreaming 2.0 规范批次。"""
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = tuple(reader.fieldnames or ())
        if actual not in (MEMORY_CSV_COLUMNS, TIMESTAMPED_MEMORY_CSV_COLUMNS):
            raise SchemaError(
                "memory CSV columns must be exactly the compact five columns, "
                "optionally followed by created_at,updated_at"
            )
        source_rows = []
        for row_number, row in enumerate(reader, 2):
            row["__row_number"] = str(row_number)
            source_rows.append(row)
        memories = []
        for row in preprocess_memory_rows(source_rows):
            row_number = int(row.pop("__row_number"))
            source_type = row["type"].strip().casefold()
            try:
                mem_type = _TYPE_ALIASES[source_type]
            except KeyError as exc:
                raise SchemaError(
                    f"memory CSV row {row_number} has unsupported type: {row['type']!r}"
                ) from exc
            topic = row["topic"].strip() or None
            subtopic = row["subtopic"].strip() or None
            if topic is None:
                subtopic = None
            memories.append(
                {
                    "memory_id": row["id"].strip(),
                    "mem_type": mem_type,
                    "content": row["content"].strip(),
                    "source_memory_ids": [],
                    "is_important": False,
                    "source_session_ids": [],
                    "created_at": None,
                    "topic": topic,
                    "subtopic": subtopic,
                }
            )
    return MemoryBatch.from_dict({"schema_version": "2.0", "memories": memories})


def memory_batch_to_csv_text(batch: MemoryBatch) -> str:
    """将规范批次序列化为紧凑 CSV，并有意省略血缘字段。"""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=MEMORY_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for memory in batch.memories:
        writer.writerow(
            {
                "id": memory.memory_id,
                "type": _CSV_TYPES[memory.mem_type.value],
                "topic": memory.topic or "",
                "subtopic": memory.subtopic or "",
                "content": memory.content,
            }
        )
    return output.getvalue()


def write_memory_batch_csv(path: str | Path, batch: MemoryBatch) -> None:
    Path(path).write_text("\ufeff" + memory_batch_to_csv_text(batch), encoding="utf-8")
