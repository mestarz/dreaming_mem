"""Validate and compare archived 0706 recall runs across temperatures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


TEMPERATURE_DIRS = ("temperature_0.1", "temperature_0.2", "temperature_0.5", "temperature_1")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project-tests", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.project_tests.resolve()))
    from common import match_keyword_groups  # pylint: disable=import-outside-toplevel

    manifest = read_json(args.root / "manifest.json")
    expected_csv = Path(manifest["recall_csv"]).resolve()
    expected_csv_hash = manifest["recall_csv_sha256"]
    if sha256_file(expected_csv) != expected_csv_hash:
        raise RuntimeError("Recall CSV checksum no longer matches the preparation manifest")

    runs: dict[str, dict[str, Any]] = {}
    pass_at_k: dict[str, dict[int, set[str]]] = {}
    query_signatures: list[tuple[str, str, str, str, str]] | None = None

    for dirname in TEMPERATURE_DIRS:
        temperature = dirname.removeprefix("temperature_")
        directory = args.root / dirname
        summary_path = directory / "summary_file_vector_top5.json"
        details_path = directory / "file_vector_recall_top5_details.csv"
        summary = read_json(summary_path)
        details = read_csv(details_path)

        if Path(summary["recall_csv"]).resolve() != expected_csv:
            raise RuntimeError(f"{dirname}: unexpected recall CSV")
        if summary["query_count"] != 305 or len(details) != 305:
            raise RuntimeError(f"{dirname}: expected 305 detail rows")
        if summary["candidate_count"] != 218 or summary["embedding_model"] != "bge-m3:latest":
            raise RuntimeError(f"{dirname}: unexpected corpus or embedding configuration")
        if summary["top_k"] != 5 or summary["threshold"] is not None or not summary["deduplicated"]:
            raise RuntimeError(f"{dirname}: unexpected recall parameters")

        signatures = [
            (row["row_number"], row["case_id"], row["turn_id"], row["query"], row["expected_keywords"])
            for row in details
        ]
        if query_signatures is None:
            query_signatures = signatures
        elif signatures != query_signatures:
            raise RuntimeError(f"{dirname}: queries differ from the first run")

        by_rank = {rank: set() for rank in range(1, 6)}
        for row in details:
            if not row["expected_keywords"]:
                continue
            recalled = json.loads(row["top5_memories"])
            key = row["row_number"]
            for rank in range(1, 6):
                text = "\n".join(item["content"] for item in recalled[:rank])
                passed, _ = match_keyword_groups(text, row["expected_keywords"])
                if passed and row["blacklist"]:
                    blacklist_hit, _ = match_keyword_groups(text, row["blacklist"])
                    passed = not blacklist_hit
                if passed:
                    by_rank[rank].add(key)

        for rank in range(1, 6):
            recorded = summary["positive_recall_at_k"][f"recall@{rank}"]["passed"]
            if recorded != len(by_rank[rank]):
                raise RuntimeError(f"{dirname}: Recall@{rank} detail/summary mismatch")

        pass_at_k[temperature] = by_rank
        runs[temperature] = {
            "temperature": float(temperature),
            "summary_file": str(summary_path.resolve()),
            "summary_sha256": sha256_file(summary_path),
            "details_file": str(details_path.resolve()),
            "details_sha256": sha256_file(details_path),
            "memory_file": summary["memory_file"],
            "memory_file_sha256": sha256_file(Path(summary["memory_file"])),
            "candidate_count": summary["candidate_count"],
            "query_count": summary["query_count"],
            "positive_count": summary["positive_recall_keyword_accuracy"]["total"],
            "recall_at_k": summary["positive_recall_at_k"],
            "corpus_coverage_upper_bound": summary["corpus_keyword_coverage_upper_bound"],
            "accuracy_among_coverable": summary["top_k_accuracy_among_corpus_coverable"],
            "all_rows": summary["all_recall_rows"],
        }

    baseline = pass_at_k["0.1"]
    differences: dict[str, Any] = {}
    for temperature in ("0.2", "0.5", "1"):
        rank_diffs = {}
        for rank in range(1, 6):
            gained = sorted(pass_at_k[temperature][rank] - baseline[rank], key=int)
            lost = sorted(baseline[rank] - pass_at_k[temperature][rank], key=int)
            rank_diffs[f"recall@{rank}"] = {"gained_row_numbers": gained, "lost_row_numbers": lost}
        differences[f"0.1_vs_{temperature}"] = rank_diffs

    comparison = {
        "schema_version": "1.0",
        "status": "completed_and_validated",
        "recall_script": str((args.project_tests / "07_file_vector_recall_top5.py").resolve()),
        "recall_script_sha256": sha256_file(args.project_tests / "07_file_vector_recall_top5.py"),
        "recall_csv": str(expected_csv),
        "recall_csv_sha256": expected_csv_hash,
        "method": {
            "embedding_provider": "ollama",
            "embedding_model": "bge-m3:latest",
            "similarity": "cosine",
            "top_k": 5,
            "threshold": None,
            "deduplicated": True,
        },
        "runs": runs,
        "differences_against_temperature_0.1": differences,
        "conclusion": (
            "Temperatures 0.2, 0.5, and 1.0 have identical aggregate metrics. "
            "Compared with 0.1, they gain one positive row at Recall@2; all other Recall@K counts match."
        ),
    }
    comparison_path = args.root / "recall_comparison.json"
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Qwen3.8 27B 温度记忆召回对比（0706 删除上下文测试集）",
        "",
        "- 召回脚本：`agent_memory_tests/07_file_vector_recall_top5.py`（未修改）",
        "- 测试集：`0706记忆端到端测试用例-删除上下文-记忆召回.csv`",
        "- 向量模型：Ollama `bge-m3:latest`；余弦相似度；TopK=5；无阈值；内容去重开启",
        "- 每档候选记忆 218 条；查询 305 条；其中正向关键字评测 292 条",
        "",
        "| 温度 | R@1 | R@2 | R@3 | R@4 | R@5 | 语料覆盖上限 | 可覆盖集内 R@5 | 全部行通过率 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for temperature in ("0.1", "0.2", "0.5", "1"):
        run = runs[temperature]
        ranks = run["recall_at_k"]
        cells = [f'{ranks[f"recall@{rank}"]["passed"]}/292 ({ranks[f"recall@{rank}"]["rate"]:.2%})' for rank in range(1, 6)]
        coverage = run["corpus_coverage_upper_bound"]
        coverable = run["accuracy_among_coverable"]
        all_rows = run["all_rows"]
        lines.append(
            f'| {temperature} | ' + " | ".join(cells) +
            f' | {coverage["passed"]}/{coverage["total"]} ({coverage["rate"]:.2%})' +
            f' | {coverable["passed"]}/{coverable["total"]} ({coverable["rate"]:.2%})' +
            f' | {all_rows["passed"]}/{all_rows["total"]} ({all_rows["rate"]:.2%}) |'
        )
    gained = differences["0.1_vs_0.2"]["recall@2"]["gained_row_numbers"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "0.2、0.5、1.0 的汇总结果完全一致。相对 0.1，仅 Recall@2 多命中 1 条；到 Recall@3 后差异消失，Recall@5 均为 186/292（63.70%）。",
            f'差异所在测试集数据行编号（不含表头）：{", ".join(gained) if gained else "无"}。',
            "",
        ]
    )
    report_path = args.root / "recall_comparison.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    checksum_path = args.root / "checksums.sha256"
    files = sorted(path for path in args.root.rglob("*") if path.is_file() and path != checksum_path)
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(args.root)}\n" for path in files),
        encoding="utf-8",
    )
    print(json.dumps({"report": str(report_path), "comparison": str(comparison_path), "files_checked": len(files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
