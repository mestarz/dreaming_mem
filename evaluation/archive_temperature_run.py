"""Turn one completed temperature run into a checksummed recall-ready archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    required = [
        "input_memories.json",
        "dreaming_result.json",
        "raw_llm_responses.json",
        "comparison.json",
        "run_metadata.json",
        "report.md",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise SystemExit("cannot archive incomplete run; missing: " + ", ".join(missing))

    metadata = read_json(run_dir / "run_metadata.json")
    result = read_json(run_dir / "dreaming_result.json")
    temperature = float(metadata["config"]["temperature"])
    model = metadata["model"]
    records = []
    for item in result["memories"]:
        records.append(
            {
                "mem_id": item["memory_id"],
                "content": item["content"],
                "type": item["mem_type"],
                "timestamp": metadata["started_at"],
                "source_id": f"standalone_dreaming_temperature_{temperature:g}",
                "source_memory_ids": item["source_memory_ids"],
                "is_important": item["is_important"],
                "model": model,
                "temperature": temperature,
                "top_p": metadata["config"].get("top_p"),
            }
        )
    write_json(run_dir / "recall_corpus.json", records)
    with (run_dir / "recall_corpus.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = [
            "mem_id",
            "content",
            "type",
            "timestamp",
            "source_id",
            "source_memory_ids",
            "is_important",
            "model",
            "temperature",
            "top_p",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["source_memory_ids"] = json.dumps(row["source_memory_ids"], ensure_ascii=False)
            writer.writerow(row)

    archive_readme = f"""# 独立 Dreaming 召回测试存档

- 模型：`{model}`
- 温度：`{temperature:g}`
- top_p：`{metadata['config'].get('top_p')}`
- 输入记忆：{metadata['input_count']} 条
- 二次萃取结果：{len(records)} 条

后续单独召回测试优先使用 `recall_corpus.json`；需要表格工具时使用
`recall_corpus.csv`。`dreaming_result.json` 是独立模块的原生输出，
`input_memories.json` 是本次完整输入，`raw_llm_responses.json` 用于审计模型原文。

运行召回测试前可执行：

```bash
sha256sum -c checksums.sha256
```
"""
    (run_dir / "ARCHIVE_README.md").write_text(archive_readme, encoding="utf-8")
    optional_audit_files = [name for name in ("batch_results.json", "group_progress.json") if (run_dir / name).is_file()]
    manifest = {
        "schema_version": "1.0",
        "status": "completed",
        "dataset_type": "standalone_dreaming_recall_corpus",
        "model": model,
        "temperature": temperature,
        "top_p": metadata["config"].get("top_p"),
        "record_count": len(records),
        "record_types": dict(sorted(Counter(row["type"] for row in records).items())),
        "important_count": sum(bool(row["is_important"]) for row in records),
        "input_count": metadata["input_count"],
        "input_source_count": metadata["input_source_count"],
        "omitted_count": metadata["omitted_count"],
        "started_at": metadata["started_at"],
        "elapsed_seconds": metadata["elapsed_seconds"],
        "source_files": {
            "first_pass_memories": metadata["input_source"],
            "original_dreaming_after": metadata["baseline_after"],
            "original_dreaming_added": metadata["baseline_added"],
        },
        "recall_files": ["recall_corpus.json", "recall_corpus.csv"],
        "audit_files": required + optional_audit_files + ["ARCHIVE_README.md"],
    }
    write_json(run_dir / "archive_manifest.json", manifest)

    checksum_files = sorted(path for path in run_dir.iterdir() if path.is_file() and path.name != "checksums.sha256")
    checksum_lines = [f"{sha256_file(path)}  {path.name}" for path in checksum_files]
    (run_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
