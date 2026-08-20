"""Reproduce standalone dreaming over Agent Memory's first-pass memories."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_dreaming import DreamingConfig, DreamingExtractor, MemoryBatch, OllamaChatLLM


DEFAULT_BASELINE_DIR = (
    WORKSPACE_ROOT
    / "agent_memory_tests"
    / "runtimes"
    / "full_0706_dreaming_batch2_20260818"
    / "dreaming_results"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "qwen3_8-27b-agent-memory-20260819"


@dataclass(slots=True)
class RecordingLLM:
    client: OllamaChatLLM
    responses: list[str] = field(default_factory=list)

    async def complete(self, prompt: str) -> str:
        response = await self.client.complete(prompt)
        self.responses.append(response)
        return response


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def convert_input(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "batch_id": "full_0706_first_pass_memories",
        "user_id": summary["user_id"],
        "scope_id": summary["scope_id"],
        "memories": [
            {
                "memory_id": row["mem_id"],
                "mem_type": row["type"],
                "content": row["content"],
                "source_session_id": row.get("source_id"),
                "created_at": row.get("timestamp"),
                "is_important": bool(row.get("is_important", False)),
            }
            for row in rows
        ],
    }


def normalized(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text).lower()


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalized(left), normalized(right)).ratio()


def nearest_matches(outputs: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for output in outputs:
        candidates = [row for row in baseline if row["type"] == output["mem_type"]]
        if not candidates:
            matches.append(
                {
                    "output": output,
                    "baseline": None,
                    "similarity": 0.0,
                    "exact_normalized": False,
                }
            )
            continue
        best = max(candidates, key=lambda row: similarity(output["content"], row["content"]))
        score = similarity(output["content"], best["content"])
        matches.append(
            {
                "output": output,
                "baseline": {
                    "mem_id": best["mem_id"],
                    "type": best["type"],
                    "content": best["content"],
                    "source_id": best.get("source_id"),
                },
                "similarity": round(score, 4),
                "exact_normalized": normalized(output["content"]) == normalized(best["content"]),
            }
        )
    return matches


def type_counts(rows: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    return dict(sorted(Counter(row[field_name] for row in rows).items()))


def build_report(metadata: dict[str, Any], comparison: dict[str, Any]) -> str:
    lines = [
        "# 独立 Dreaming × qwen3.8:27b 实测报告",
        "",
        "## 运行信息",
        "",
        f"- 模型：`{metadata['model']}`",
        f"- 执行时间：{metadata['started_at']}，耗时 {metadata['elapsed_seconds']} 秒",
        f"- 输入：Agent Memory 第一次直接萃取结果，共 {metadata['input_count']} 条",
        f"- 输入来源会话数：{metadata['input_source_count']} 个",
        f"- 独立模块输出：{metadata['output_count']} 条",
        f"- 因 Token 预算省略：{metadata['omitted_count']} 条",
        f"- 模型调用次数（含格式重试）：{metadata['llm_call_count']} 次",
        "",
        "## 数量与类型",
        "",
        "| 数据集 | 总数 | user_profile | episodic_memory | semantic_memory |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison["count_table"]:
        counts = row["types"]
        lines.append(
            f"| {row['name']} | {row['count']} | {counts.get('user_profile', 0)} | "
            f"{counts.get('episodic_memory', 0)} | {counts.get('semantic_memory', 0)} |"
        )
    lines.extend(
        [
            "",
            "## 与原 Dreaming 新增记忆的文本对照",
            "",
            f"- 归一化文本完全一致：{comparison['exact_match_count']} 条",
            f"- 最佳相似度 ≥ 0.80：{comparison['similarity_ge_080_count']} 条",
            f"- 最佳相似度 ≥ 0.60：{comparison['similarity_ge_060_count']} 条",
            f"- 平均最佳相似度：{comparison['average_best_similarity']}",
            "",
            "这里的相似度是字符级诊断指标，不等同于语义正确率。原 Agent Memory Dreaming 的输入是原始会话、逐会话最多输出 5 条并经过存储去重；独立模块的输入是全部第一次萃取记忆，一次做全局整合，因此不应期待逐条完全一致。",
            "",
            "## 最接近的结果（按相似度排序）",
            "",
        ]
    )
    for index, match in enumerate(comparison["nearest_matches"][:20], 1):
        baseline = match["baseline"]
        lines.append(f"### {index}. 相似度 {match['similarity']}")
        lines.append("")
        lines.append(f"- 独立模块：{match['output']['content']}")
        lines.append(f"- 原 Dreaming：{baseline['content'] if baseline else '无同类型候选'}")
        lines.append(f"- 来源记忆：{', '.join(match['output']['source_memory_ids'])}")
        lines.append("")
    lines.extend(
        [
            "## 产物",
            "",
            "- `input_memories.json`：由 Agent Memory 第一次萃取结果转换而来的完整输入。",
            "- `dreaming_result.json`：独立模块的结构化输出。",
            "- `raw_llm_responses.json`：模型原始响应（包括可能发生的重试）。",
            "- `comparison.json`：完整对照指标和逐条最近匹配。",
            "- `run_metadata.json`：模型、耗时、输入覆盖和调用次数。",
            "",
        ]
    )
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before = read_json(args.baseline_dir / "memories_before.json")
    after = read_json(args.baseline_dir / "memories_after.json")
    added = read_json(args.baseline_dir / "dreaming_added_memories.json")
    removed = read_json(args.baseline_dir / "dreaming_removed_memories.json")
    summary = read_json(args.baseline_dir / "summary_dreaming.json")
    converted = convert_input(before, summary)
    write_json(args.output_dir / "input_memories.json", converted)

    batch = MemoryBatch.from_dict(converted)
    recorder = RecordingLLM(
        OllamaChatLLM(
            base_url=args.base_url,
            model=args.model,
            timeout_seconds=args.timeout,
            temperature=args.temperature,
            top_p=args.top_p,
            think=False,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            format_schema=None,
        )
    )
    config = DreamingConfig(
        max_input_tokens=args.max_input_tokens,
        max_output_items=args.max_output_items,
        retries=args.retries,
    )
    started_at = datetime.now().astimezone().isoformat()
    started = time.monotonic()
    try:
        result = await DreamingExtractor(recorder, config).dream(batch)
    except Exception as exc:
        write_json(args.output_dir / "raw_llm_responses.json", recorder.responses)
        write_json(
            args.output_dir / "failed_run.json",
            {
                "status": "failed",
                "model": args.model,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "think": False,
                "num_ctx": args.num_ctx,
                "num_predict": args.num_predict,
                "ollama_format": None,
                "module_max_output_items": args.max_output_items,
                "started_at": started_at,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "llm_call_count": len(recorder.responses),
            },
        )
        raise
    elapsed = round(time.monotonic() - started, 3)
    result_data = result.to_dict()
    outputs = result_data["memories"]
    write_json(args.output_dir / "dreaming_result.json", result_data)
    write_json(args.output_dir / "raw_llm_responses.json", recorder.responses)

    matches = nearest_matches(outputs, added)
    matches.sort(key=lambda item: item["similarity"], reverse=True)
    scores = [item["similarity"] for item in matches]
    comparison = {
        "count_table": [
            {"name": "第一次直接萃取", "count": len(before), "types": type_counts(before, "type")},
            {"name": "原 Dreaming 新增", "count": len(added), "types": type_counts(added, "type")},
            {"name": "原 Dreaming 删除", "count": len(removed), "types": type_counts(removed, "type")},
            {"name": "原 Dreaming 最终记忆", "count": len(after), "types": type_counts(after, "type")},
            {"name": "独立模块二次萃取", "count": len(outputs), "types": type_counts(outputs, "mem_type")},
        ],
        "exact_match_count": sum(item["exact_normalized"] for item in matches),
        "similarity_ge_080_count": sum(item["similarity"] >= 0.8 for item in matches),
        "similarity_ge_060_count": sum(item["similarity"] >= 0.6 for item in matches),
        "average_best_similarity": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "nearest_matches": matches,
    }
    metadata = {
        "model": args.model,
        "base_url": args.base_url,
        "started_at": started_at,
        "elapsed_seconds": elapsed,
        "input_source": str(args.baseline_dir / "memories_before.json"),
        "baseline_after": str(args.baseline_dir / "memories_after.json"),
        "baseline_added": str(args.baseline_dir / "dreaming_added_memories.json"),
        "input_count": len(before),
        "input_source_count": len({row.get("source_id") for row in before}),
        "output_count": len(outputs),
        "omitted_count": len(result.omitted_memory_ids),
        "llm_call_count": len(recorder.responses),
        "config": {
            "max_input_tokens": args.max_input_tokens,
            "max_output_items": args.max_output_items,
            "retries": args.retries,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "think": False,
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
            "ollama_format": None,
            "module_max_output_items": args.max_output_items,
            "timeout": args.timeout,
        },
    }
    write_json(args.output_dir / "comparison.json", comparison)
    write_json(args.output_dir / "run_metadata.json", metadata)
    (args.output_dir / "report.md").write_text(build_report(metadata, comparison), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.8:27b")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.1)
    parser.add_argument("--num-ctx", type=int, default=131_072)
    parser.add_argument("--num-predict", type=int, default=65_536)
    parser.add_argument("--max-input-tokens", type=int, default=100_000)
    parser.add_argument("--max-output-items", type=int, default=100)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
