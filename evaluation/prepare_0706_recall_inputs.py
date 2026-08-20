"""Convert archived dreaming recall corpora to the project's 11-column TSV format."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--recall-csv", type=Path, required=True)
    parser.add_argument("--temperatures", nargs="+", type=float, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    datasets = []
    for temperature in args.temperatures:
        source = args.archive_root / f"temperature_{temperature:g}" / "recall_corpus.json"
        records = read_json(source)
        output_dir = args.output_root / f"temperature_{temperature:g}"
        output_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = output_dir / "memory_corpus.tsv"
        with tsv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            for record in records:
                extra = json.dumps(
                    {
                        "source_memory_ids": record["source_memory_ids"],
                        "is_important": record["is_important"],
                        "model": record["model"],
                        "temperature": record["temperature"],
                        "top_p": record["top_p"],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
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
                        "standalone_dreaming_recall_user",
                        extra,
                    ]
                )
        datasets.append(
            {
                "temperature": temperature,
                "record_count": len(records),
                "source": str(source.resolve()),
                "source_sha256": sha256_file(source),
                "memory_file": str(tsv_path.resolve()),
                "memory_file_sha256": sha256_file(tsv_path),
                "output_dir": str(output_dir.resolve()),
            }
        )
    manifest = {
        "schema_version": "1.0",
        "status": "prepared",
        "project_recall_script": "agent_memory_tests/07_file_vector_recall_top5.py",
        "recall_csv": str(args.recall_csv.resolve()),
        "recall_csv_sha256": sha256_file(args.recall_csv),
        "embedding_model": "bge-m3:latest",
        "top_k": 5,
        "threshold": None,
        "deduplicate": True,
        "datasets": datasets,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
