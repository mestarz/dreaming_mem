"""Convert the archived pre-Dreaming extraction into the recall TSV format."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recall-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a non-empty JSON list")
    required = {"mem_id", "content", "type", "timestamp", "source_id"}
    for index, record in enumerate(records, 1):
        missing = required - set(record)
        if missing:
            raise ValueError(f"Record {index} is missing fields: {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    memory_file = args.output_dir / "memory_corpus.tsv"
    with memory_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        for record in records:
            writer.writerow(
                [
                    record["timestamp"],
                    record["content"],
                    "[0.0]",
                    record["mem_id"],
                    r"\N",
                    "1",
                    record["type"],
                    r"\N",
                    record["timestamp"],
                    "pre_dreaming_extraction_user",
                    json.dumps(
                        {"source_id": record["source_id"], "stage": "pre_dreaming_extraction"},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ]
            )

    manifest = {
        "schema_version": "1.0",
        "stage": "pre_dreaming_direct_extraction",
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "source_record_count": len(records),
        "source_types": dict(sorted(Counter(record["type"] for record in records).items())),
        "memory_file": str(memory_file.resolve()),
        "memory_file_sha256": sha256_file(memory_file),
        "recall_csv": str(args.recall_csv.resolve()),
        "recall_csv_sha256": sha256_file(args.recall_csv),
        "recall_configuration": {
            "script": "agent_memory_tests/07_file_vector_recall_top5.py",
            "embedding_model": "bge-m3:latest",
            "top_k": 5,
            "threshold": None,
            "deduplicate": True,
        },
    }
    manifest_path = args.output_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
