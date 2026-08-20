"""Compare pre-Dreaming direct extraction recall with each Dreaming corpus."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


TEMPERATURES = ("0.1", "0.2", "0.5", "1")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def evaluate_at_k(rows: list[dict[str, str]], match_keyword_groups: Any) -> dict[int, set[str]]:
    passed_by_rank = {rank: set() for rank in range(1, 6)}
    for row in rows:
        if not row["expected_keywords"]:
            continue
        recalled = json.loads(row["top5_memories"])
        for rank in range(1, 6):
            recalled_text = "\n".join(item["content"] for item in recalled[:rank])
            passed, _ = match_keyword_groups(recalled_text, row["expected_keywords"])
            if passed and row["blacklist"]:
                blacklist_hit, _ = match_keyword_groups(recalled_text, row["blacklist"])
                passed = not blacklist_hit
            if passed:
                passed_by_rank[rank].add(row["row_number"])
    return passed_by_rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project-tests", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project_tests.resolve()))
    from common import match_keyword_groups  # pylint: disable=import-outside-toplevel

    baseline_dir = args.root / "pre_dreaming"
    baseline_summary = read_json(baseline_dir / "summary_file_vector_top5.json")
    baseline_rows = read_csv(baseline_dir / "file_vector_recall_top5_details.csv")
    baseline_pass = evaluate_at_k(baseline_rows, match_keyword_groups)
    baseline_queries = [(row["row_number"], row["query"], row["expected_keywords"]) for row in baseline_rows]

    if baseline_summary["candidate_count"] != 438 or len(baseline_rows) != 305:
        raise RuntimeError("Unexpected pre-Dreaming baseline shape")

    comparisons: dict[str, Any] = {}
    for temperature in TEMPERATURES:
        directory = args.root / f"temperature_{temperature}"
        summary = read_json(directory / "summary_file_vector_top5.json")
        rows = read_csv(directory / "file_vector_recall_top5_details.csv")
        queries = [(row["row_number"], row["query"], row["expected_keywords"]) for row in rows]
        if queries != baseline_queries:
            raise RuntimeError(f"Temperature {temperature}: query rows differ from baseline")
        dreaming_pass = evaluate_at_k(rows, match_keyword_groups)
        ranks = {}
        for rank in range(1, 6):
            before = baseline_summary["positive_recall_at_k"][f"recall@{rank}"]
            after = summary["positive_recall_at_k"][f"recall@{rank}"]
            improved = sorted(dreaming_pass[rank] - baseline_pass[rank], key=int)
            regressed = sorted(baseline_pass[rank] - dreaming_pass[rank], key=int)
            ranks[f"recall@{rank}"] = {
                "before": before,
                "after": after,
                "passed_delta": after["passed"] - before["passed"],
                "delta_percentage_points": round((after["rate"] - before["rate"]) * 100, 4),
                "improved_count": len(improved),
                "regressed_count": len(regressed),
                "improved_row_numbers": improved,
                "regressed_row_numbers": regressed,
            }
        before_coverage = baseline_summary["corpus_keyword_coverage_upper_bound"]
        after_coverage = summary["corpus_keyword_coverage_upper_bound"]
        before_coverable = baseline_summary["top_k_accuracy_among_corpus_coverable"]
        after_coverable = summary["top_k_accuracy_among_corpus_coverable"]
        comparisons[temperature] = {
            "pre_dreaming_candidate_count": baseline_summary["candidate_count"],
            "post_dreaming_candidate_count": summary["candidate_count"],
            "recall_at_k": ranks,
            "corpus_coverage": {
                "before": before_coverage,
                "after": after_coverage,
                "passed_delta": after_coverage["passed"] - before_coverage["passed"],
                "delta_percentage_points": round((after_coverage["rate"] - before_coverage["rate"]) * 100, 4),
            },
            "accuracy_among_coverable": {
                "before": before_coverable,
                "after": after_coverable,
                "delta_percentage_points": round((after_coverable["rate"] - before_coverable["rate"]) * 100, 4),
            },
        }

    output = {
        "schema_version": "1.0",
        "status": "completed_and_validated",
        "baseline": "pre_dreaming_direct_extraction",
        "test_set": baseline_summary["recall_csv"],
        "method": {
            "embedding_model": "bge-m3:latest",
            "similarity": "cosine",
            "top_k": 5,
            "threshold": None,
            "deduplicated": True,
        },
        "comparisons": comparisons,
    }
    json_path = args.root / "pre_vs_dreaming_comparison.json"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Dreaming 前后召回对比",
        "",
        "同一 0706 删除上下文召回测试集、同一项目文件召回脚本及 `bge-m3:latest` 向量模型。",
        "",
        "| 阶段/温度 | 候选记忆 | R@1 | R@2 | R@3 | R@4 | R@5 | 语料覆盖上限 | 可覆盖集内 R@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    before_ranks = baseline_summary["positive_recall_at_k"]
    before_coverage = baseline_summary["corpus_keyword_coverage_upper_bound"]
    before_coverable = baseline_summary["top_k_accuracy_among_corpus_coverable"]
    cells = [f'{before_ranks[f"recall@{rank}"]["passed"]}/292 ({before_ranks[f"recall@{rank}"]["rate"]:.2%})' for rank in range(1, 6)]
    lines.append(
        "| Dreaming 前 | 438 | " + " | ".join(cells) +
        f' | {before_coverage["passed"]}/292 ({before_coverage["rate"]:.2%})' +
        f' | {before_coverable["passed"]}/{before_coverable["total"]} ({before_coverable["rate"]:.2%}) |'
    )
    for temperature in TEMPERATURES:
        summary = read_json(args.root / f"temperature_{temperature}" / "summary_file_vector_top5.json")
        ranks = summary["positive_recall_at_k"]
        coverage = summary["corpus_keyword_coverage_upper_bound"]
        coverable = summary["top_k_accuracy_among_corpus_coverable"]
        cells = [f'{ranks[f"recall@{rank}"]["passed"]}/292 ({ranks[f"recall@{rank}"]["rate"]:.2%})' for rank in range(1, 6)]
        lines.append(
            f"| Dreaming 后 {temperature} | 218 | " + " | ".join(cells) +
            f' | {coverage["passed"]}/292 ({coverage["rate"]:.2%})' +
            f' | {coverable["passed"]}/{coverable["total"]} ({coverable["rate"]:.2%}) |'
        )

    r01 = comparisons["0.1"]["recall_at_k"]
    coverage01 = comparisons["0.1"]["corpus_coverage"]
    coverable01 = comparisons["0.1"]["accuracy_among_coverable"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f'Dreaming 后 0.1 的 R@1 提升 {r01["recall@1"]["passed_delta"]} 条（{r01["recall@1"]["delta_percentage_points"]:+.2f} 个百分点），R@5 提升 {r01["recall@5"]["passed_delta"]} 条（{r01["recall@5"]["delta_percentage_points"]:+.2f} 个百分点）。',
            f'R@5 的逐条变化为改善 {r01["recall@5"]["improved_count"]} 条、退化 {r01["recall@5"]["regressed_count"]} 条，净提升 {r01["recall@5"]["passed_delta"]} 条。',
            f'语料覆盖上限变化 {coverage01["passed_delta"]} 条（{coverage01["delta_percentage_points"]:+.2f} 个百分点），但可覆盖集内 R@5 提升 {coverable01["delta_percentage_points"]:+.2f} 个百分点。',
            "",
        ]
    )
    markdown_path = args.root / "pre_vs_dreaming_comparison.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "report": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
