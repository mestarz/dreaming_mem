"""Compare completed qwen3.8 temperature runs using deterministic text metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text).lower()


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalized(left), normalized(right)).ratio()


def directional_best(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    if not left or not right:
        return 0.0
    values = []
    for item in left:
        same_type = [candidate for candidate in right if candidate["mem_type"] == item["mem_type"]]
        values.append(max((similarity(item["content"], row["content"]) for row in same_type), default=0.0))
    return sum(values) / len(values)


def baseline_metrics(
    outputs: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    baseline_type_field: str,
) -> dict[str, Any]:
    normalized_baseline = [
        {"mem_type": row[baseline_type_field], "content": row["content"]} for row in baseline
    ]
    baseline_texts = {(row["mem_type"], normalized(row["content"])) for row in normalized_baseline}
    best_scores = []
    for output in outputs:
        candidates = [row for row in normalized_baseline if row["mem_type"] == output["mem_type"]]
        best_scores.append(
            max((similarity(output["content"], candidate["content"]) for candidate in candidates), default=0.0)
        )
    return {
        "exact_match_count": sum(
            (row["mem_type"], normalized(row["content"])) in baseline_texts for row in outputs
        ),
        "similarity_ge_080_count": sum(score >= 0.8 for score in best_scores),
        "similarity_ge_060_count": sum(score >= 0.6 for score in best_scores),
        "average_best_similarity": round(sum(best_scores) / len(best_scores), 4) if best_scores else 0.0,
    }


def full_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["mem_type"],
        normalized(row["content"]),
        tuple(row["source_memory_ids"]),
        bool(row["is_important"]),
    )


def compare_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rows = left["rows"]
    right_rows = right["rows"]
    left_texts = {(row["mem_type"], normalized(row["content"])) for row in left_rows}
    right_texts = {(row["mem_type"], normalized(row["content"])) for row in right_rows}
    left_by_text = {(row["mem_type"], normalized(row["content"])): row for row in left_rows}
    right_by_text = {(row["mem_type"], normalized(row["content"])): row for row in right_rows}
    union = left_texts | right_texts
    left_full = {full_signature(row) for row in left_rows}
    right_full = {full_signature(row) for row in right_rows}
    return {
        "left_temperature": left["temperature"],
        "right_temperature": right["temperature"],
        "exact_overlap_count": len(left_texts & right_texts),
        "full_record_overlap_count": len(left_full & right_full),
        "exact_jaccard": round(len(left_texts & right_texts) / len(union), 4) if union else 1.0,
        "average_best_similarity_left_to_right": round(directional_best(left_rows, right_rows), 4),
        "average_best_similarity_right_to_left": round(directional_best(right_rows, left_rows), 4),
        "unique_to_left": [left_by_text[key] for key in sorted(left_texts - right_texts)],
        "unique_to_right": [right_by_text[key] for key in sorted(right_texts - left_texts)],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--temperatures", nargs="+", type=float, required=True)
    args = parser.parse_args()

    runs = []
    for temperature in args.temperatures:
        run_dir = args.root / f"temperature_{temperature:g}"
        metadata = read_json(run_dir / "run_metadata.json")
        result = read_json(run_dir / "dreaming_result.json")
        input_rows = read_json(run_dir / "input_memories.json")["memories"]
        original_added = read_json(Path(metadata["baseline_added"]))
        original_after = read_json(Path(metadata["baseline_after"]))
        rows = result["memories"]
        runs.append(
            {
                "temperature": temperature,
                "directory": str(run_dir),
                "rows": rows,
                "count": len(rows),
                "types": dict(sorted(Counter(row["mem_type"] for row in rows).items())),
                "important_count": sum(bool(row["is_important"]) for row in rows),
                "elapsed_seconds": metadata["elapsed_seconds"],
                "llm_call_count": metadata["llm_call_count"],
                "against_first_pass": baseline_metrics(rows, input_rows, "mem_type"),
                "against_original_dreaming_added": baseline_metrics(rows, original_added, "type"),
                "against_original_dreaming_after": baseline_metrics(rows, original_after, "type"),
            }
        )
    pairs = [compare_pair(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))]
    serializable_runs = [{key: value for key, value in run.items() if key != "rows"} for run in runs]
    output = {"runs": serializable_runs, "pairwise": pairs}
    (args.root / "temperature_comparison.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# qwen3.8:27b 温度对照",
        "",
        "所有运行使用相同 438 条第一次萃取记忆、相同提示词和 `top_p=0.1`，仅改变温度。",
        "",
        "执行方式：按第一次萃取记录的 103 个来源会话分组，每组最多输出 5 条，然后跨组精确去重。",
        "",
        "| 温度 | 输出数 | 重要 | user_profile | episodic | semantic | 耗时(s) | 对第一次：完全/平均 | 对原新增：完全/平均 | 对原最终：完全/平均 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        types = run["types"]
        lines.append(
            f"| {run['temperature']:g} | {run['count']} | {run['important_count']} | "
            f"{types.get('user_profile', 0)} | {types.get('episodic_memory', 0)} | "
            f"{types.get('semantic_memory', 0)} | {run['elapsed_seconds']} | "
            f"{run['against_first_pass']['exact_match_count']} / {run['against_first_pass']['average_best_similarity']} | "
            f"{run['against_original_dreaming_added']['exact_match_count']} / "
            f"{run['against_original_dreaming_added']['average_best_similarity']} | "
            f"{run['against_original_dreaming_after']['exact_match_count']} / "
            f"{run['against_original_dreaming_after']['average_best_similarity']} |"
        )
    lines.extend(
        [
            "",
            "## 温度间两两对照",
            "",
            "| 温度 A | 温度 B | 文本重合 | 完整记录重合 | 精确 Jaccard | A→B 平均最佳相似度 | B→A 平均最佳相似度 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in pairs:
        lines.append(
            f"| {pair['left_temperature']:g} | {pair['right_temperature']:g} | "
            f"{pair['exact_overlap_count']} | {pair['full_record_overlap_count']} | {pair['exact_jaccard']} | "
            f"{pair['average_best_similarity_left_to_right']} | "
            f"{pair['average_best_similarity_right_to_left']} |"
        )
    lines.extend(
        [
            "",
            "字符相似度用于观察温度造成的表述与选择变化，不代表事实正确率；事实和来源可在各目录的 `dreaming_result.json` 中逐条审阅。",
            "",
        ]
    )
    (args.root / "temperature_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "status": "completed",
        "study": "qwen3.8:27b standalone dreaming temperature comparison",
        "model": "qwen3.8:27b",
        "temperatures": args.temperatures,
        "controlled_parameters": {
            "top_p": 0.1,
            "think": False,
            "group_by": "source_id from baseline records",
            "group_count": 103,
            "max_items_per_group": 5,
        },
        "input": {
            "first_pass_memory_count": 438,
            "source_session_count": 103,
        },
        "historical_baseline": {
            "model": "huihui_ai/Qwen3.6-abliterated:27b",
            "temperature": 0.2,
            "top_p": 0.1,
            "original_dreaming_added_count": 69,
            "original_dreaming_after_count": 465,
        },
        "archives": [
            {
                "temperature": run["temperature"],
                "directory": f"temperature_{run['temperature']:g}",
                "recall_corpus": f"temperature_{run['temperature']:g}/recall_corpus.json",
                "record_count": run["count"],
                "checksums": f"temperature_{run['temperature']:g}/checksums.sha256",
            }
            for run in runs
        ],
        "comparison_files": ["temperature_comparison.json", "temperature_comparison.md"],
        "diagnostics_directory": "diagnostics",
    }
    manifest_path = args.root / "study_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    study_files = [
        args.root / "temperature_comparison.json",
        args.root / "temperature_comparison.md",
        manifest_path,
        *[args.root / f"temperature_{temperature:g}" / "checksums.sha256" for temperature in args.temperatures],
    ]
    checksum_lines = [f"{sha256_file(path)}  {path.relative_to(args.root)}" for path in study_files]
    (args.root / "study_checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
