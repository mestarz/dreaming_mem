"""Dreaming 命令行接口：支持 JSON/CSV 输入与 JSON 输出。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from .engine import DreamingConfig, DreamingExtractor
from .csv_io import memory_batch_from_csv, write_memory_batch_csv
from .llm import OpenAICompatibleLLM
from .models import MemoryBatch, SchemaError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对已提取记忆执行二阶段 Dreaming")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="2.0 规范 JSON 或紧凑记忆 CSV 路径；使用 - 从标准输入读取 JSON",
    )
    parser.add_argument("--output", "-o", default="-", help="输出 JSON 路径；使用 - 输出到标准输出")
    parser.add_argument("--base-url", default=os.getenv("DREAMING_API_BASE"))
    parser.add_argument("--api-key", default=os.getenv("DREAMING_API_KEY"))
    parser.add_argument("--model", default=os.getenv("DREAMING_MODEL"))
    parser.add_argument("--max-input-tokens", type=int, default=30_000)
    parser.add_argument("--max-output-items", type=int, default=10)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="校验输入并打印提示词，不调用 LLM",
    )
    return parser


def _read_json(path: str) -> dict:
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise SchemaError("input document must be a JSON object")
    return value


def _read_batch(path: str) -> MemoryBatch:
    if path != "-" and Path(path).suffix.casefold() == ".csv":
        return memory_batch_from_csv(path)
    return MemoryBatch.from_dict(_read_json(path))


def _write_text(path: str, text: str) -> None:
    if path == "-":
        print(text)
    else:
        Path(path).write_text(text + "\n", encoding="utf-8")


async def _run(args: argparse.Namespace) -> int:
    batch = _read_batch(args.input)
    config = DreamingConfig(
        max_input_tokens=args.max_input_tokens,
        max_output_items=args.max_output_items,
        retries=args.retries,
    )
    if args.print_prompt:
        # 仅生成提示词时不需要真实 LLM，因此使用一个不应被调用的桩函数。
        async def unused(_: str) -> str:
            raise AssertionError("unused")

        from .llm import CallableLLM

        prompt, omitted = DreamingExtractor(CallableLLM(unused), config).build_prompt(batch)
        if omitted:
            print(f"omitted_memory_ids={json.dumps(omitted, ensure_ascii=False)}", file=sys.stderr)
        _write_text(args.output, prompt)
        return 0
    missing = [name for name in ("base_url", "api_key", "model") if not getattr(args, name)]
    if missing:
        raise SchemaError("missing LLM settings: " + ", ".join(missing))
    llm = OpenAICompatibleLLM(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout_seconds=args.timeout,
    )
    result = await DreamingExtractor(llm, config).dream(batch)
    if result.omitted_memory_ids:
        print(
            f"omitted_memory_ids={json.dumps(result.omitted_memory_ids, ensure_ascii=False)}",
            file=sys.stderr,
        )
    if args.output != "-" and Path(args.output).suffix.casefold() == ".csv":
        write_memory_batch_csv(args.output, MemoryBatch(result.memories))
    else:
        _write_text(args.output, json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except (OSError, json.JSONDecodeError, SchemaError, ValueError, RuntimeError) as exc:
        print(f"agent-dreaming: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
