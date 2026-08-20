"""Run standalone dreaming per first-pass source session and archive progress."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_dreaming import DreamingConfig, DreamingExtractor, MemoryBatch, OllamaChatLLM
from run_agent_memory_qwen import (
    DEFAULT_BASELINE_DIR,
    build_report,
    convert_input,
    nearest_matches,
    read_json,
    type_counts,
    write_json,
)


def group_memories(converted: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for memory in converted["memories"]:
        source_id = memory.get("source_session_id") or "__no_source__"
        groups.setdefault(source_id, []).append(memory)
    return list(groups.items())


def merge_exact(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for memory in memories:
        key = (memory["mem_type"], memory["content"])
        previous = merged.get(key)
        if previous is None:
            merged[key] = dict(memory)
            continue
        previous["source_memory_ids"] = list(
            dict.fromkeys(previous["source_memory_ids"] + memory["source_memory_ids"])
        )
        previous["is_important"] = previous["is_important"] or memory["is_important"]
    return list(merged.values())


def initial_progress(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "running",
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_items_per_group": args.max_items_per_group,
        "completed_source_ids": [],
        "batch_results": [],
        "raw_responses": [],
        "unmerged_memories": [],
        "elapsed_seconds": 0.0,
    }


def validate_progress(progress: dict[str, Any], args: argparse.Namespace) -> None:
    expected = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_items_per_group": args.max_items_per_group,
    }
    actual = {key: progress.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"resume configuration mismatch: expected {expected}, found {actual}")


async def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before = read_json(args.baseline_dir / "memories_before.json")
    after = read_json(args.baseline_dir / "memories_after.json")
    added = read_json(args.baseline_dir / "dreaming_added_memories.json")
    removed = read_json(args.baseline_dir / "dreaming_removed_memories.json")
    summary = read_json(args.baseline_dir / "summary_dreaming.json")
    converted = convert_input(before, summary)
    write_json(args.output_dir / "input_memories.json", converted)
    groups = group_memories(converted)

    progress_path = args.output_dir / "group_progress.json"
    if args.resume and progress_path.exists():
        progress = read_json(progress_path)
        validate_progress(progress, args)
    else:
        progress = initial_progress(args)
    completed = set(progress["completed_source_ids"])
    run_started_at = datetime.now().astimezone().isoformat()
    run_started = time.monotonic()

    client = OllamaChatLLM(
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout,
        temperature=args.temperature,
        top_p=args.top_p,
        think=False,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
    )
    try:
        for index, (source_id, memories) in enumerate(groups, 1):
            if source_id in completed:
                continue
            responses: list[str] = []

            async def complete(prompt: str) -> str:
                response = await client.complete(prompt)
                responses.append(response)
                return response

            from agent_dreaming import CallableLLM

            group_batch = MemoryBatch.from_dict(
                {
                    "schema_version": "1.0",
                    "batch_id": source_id,
                    "user_id": converted["user_id"],
                    "scope_id": converted["scope_id"],
                    "memories": memories,
                }
            )
            result = await DreamingExtractor(
                CallableLLM(complete),
                DreamingConfig(
                    max_input_tokens=args.max_input_tokens,
                    max_output_items=args.max_items_per_group,
                    retries=args.retries,
                ),
            ).dream(group_batch)
            result_memories = [item.to_dict() for item in result.memories]
            progress["unmerged_memories"].extend(result_memories)
            progress["raw_responses"].append({"source_id": source_id, "responses": responses})
            progress["batch_results"].append(
                {
                    "source_id": source_id,
                    "input_count": len(memories),
                    "output_count": len(result_memories),
                    "llm_call_count": len(responses),
                    "omitted_memory_ids": list(result.omitted_memory_ids),
                }
            )
            progress["completed_source_ids"].append(source_id)
            completed.add(source_id)
            progress["elapsed_seconds"] = round(
                float(progress.get("elapsed_seconds", 0.0)) + time.monotonic() - run_started,
                3,
            )
            run_started = time.monotonic()
            write_json(progress_path, progress)
            if index % 10 == 0 or index == len(groups):
                print(
                    f"temperature={args.temperature:g} groups={len(completed)}/{len(groups)} "
                    f"raw_items={len(progress['unmerged_memories'])}",
                    flush=True,
                )
    except Exception as exc:
        progress["status"] = "failed"
        progress["error_type"] = type(exc).__name__
        progress["error"] = str(exc)
        progress["elapsed_seconds"] = round(
            float(progress.get("elapsed_seconds", 0.0)) + time.monotonic() - run_started,
            3,
        )
        write_json(progress_path, progress)
        raise

    progress["status"] = "completed"
    write_json(progress_path, progress)
    outputs = merge_exact(progress["unmerged_memories"])
    result_data = {
        "schema_version": "1.0",
        "memories": outputs,
        "input_memory_ids": [memory["memory_id"] for memory in converted["memories"]],
        "omitted_memory_ids": [
            memory_id for batch in progress["batch_results"] for memory_id in batch["omitted_memory_ids"]
        ],
    }
    write_json(args.output_dir / "dreaming_result.json", result_data)
    write_json(args.output_dir / "raw_llm_responses.json", progress["raw_responses"])
    write_json(args.output_dir / "batch_results.json", progress["batch_results"])

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
    write_json(args.output_dir / "comparison.json", comparison)
    metadata = {
        "model": args.model,
        "base_url": args.base_url,
        "started_at": run_started_at,
        "elapsed_seconds": progress["elapsed_seconds"],
        "input_source": str(args.baseline_dir / "memories_before.json"),
        "baseline_after": str(args.baseline_dir / "memories_after.json"),
        "baseline_added": str(args.baseline_dir / "dreaming_added_memories.json"),
        "input_count": len(before),
        "input_source_count": len(groups),
        "group_count": len(groups),
        "completed_group_count": len(completed),
        "unmerged_output_count": len(progress["unmerged_memories"]),
        "output_count": len(outputs),
        "omitted_count": len(result_data["omitted_memory_ids"]),
        "llm_call_count": sum(batch["llm_call_count"] for batch in progress["batch_results"]),
        "config": {
            "group_by": "source_session_id",
            "max_input_tokens_per_group": args.max_input_tokens,
            "max_items_per_group": args.max_items_per_group,
            "retries": args.retries,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "think": False,
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
            "timeout": args.timeout,
        },
    }
    write_json(args.output_dir / "run_metadata.json", metadata)
    report = build_report(metadata, comparison)
    report = report.replace(
        "## 运行信息",
        "## 运行信息\n\n- 执行策略：按 `source_session_id` 分为 "
        f"{len(groups)} 组，每组最多输出 {args.max_items_per_group} 条，再做跨组精确去重。",
    )
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.8:27b")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, default=0.1)
    parser.add_argument("--num-ctx", type=int, default=131_072)
    parser.add_argument("--num-predict", type=int, default=4096)
    parser.add_argument("--max-input-tokens", type=int, default=30_000)
    parser.add_argument("--max-items-per-group", type=int, default=5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
