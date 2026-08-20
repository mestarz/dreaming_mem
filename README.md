# Agent Dreaming（独立记忆二次萃取模块）

本目录把 `agent-memory` 的 Dreaming 核心抽成一个可独立安装、一次调用完成的 Python 包。它接收**已经萃取过的记忆**，通过原 Dreaming 的长期价值筛选、压缩、跨条目归纳、类型约束、重要性判断和失败重试逻辑，输出可追溯的二次萃取结果。

它有意不包含原模块的会话读取、KV checkpoint、向量库存储、后台定时器和遗忘曲线。调用方决定输入来自哪里、结果如何去重与回写。因此本包没有 Agent Memory 运行时依赖，也没有第三方 Python 依赖。

## 统一记忆格式（schema 2.0）

顶层字段：

- `schema_version`：必填，固定为 `"2.0"`。
- `memories`：必填数组；允许空数组。

每条记忆字段：

- `memory_id`：必填，在批次内唯一。二次萃取结果由模块根据类型和正文生成稳定 ID。
- `mem_type`：必填，只能是 `user_profile`、`semantic_memory`、`episodic_memory`。
- `content`：必填，已经完成第一次萃取的记忆正文。
- `source_memory_ids`：必填数组；首次输入通常为 `[]`，二次结果记录其根来源记忆 ID。
- `is_important`：必填布尔值。
- `source_session_ids`：可选来源会话 ID 数组；输出会合并全部来源会话并跨轮次保留。
- `created_at`：可选、带时区的 ISO 8601 时间；输出继承其来源中的最新时间。

输入和输出使用完全相同的顶层结构与记忆字段，因此 `result.to_dict()` 可以直接传给下一次 `dream()`。未知字段会被拒绝，避免拼错或无效参数被静默忽略。schema 1.0 中没有参与萃取的 `user_id`、`scope_id`、`batch_id` 和 `metadata` 已删除。

从 schema 1.0 迁移时，需要删除上述四个字段，将 `source_session_id` 改为 `source_session_ids` 数组，并为首次萃取记录补上 `source_memory_ids: []`。这是破坏性升级，对应包版本 `0.2.0`。

完整输入见 [examples/memories.json](examples/memories.json)。

## Python 调用

```python
import asyncio
from agent_dreaming import DreamingExtractor, MemoryBatch, OpenAICompatibleLLM

payload = {
    "schema_version": "2.0",
    "memories": [
        {
            "memory_id": "mem-001",
            "mem_type": "user_profile",
            "content": "用户在项目中优先选择 Python。",
            "source_memory_ids": [],
            "is_important": False,
            "source_session_ids": ["session-101"],
            "created_at": "2026-08-01T10:00:00+08:00"
        },
        {
            "memory_id": "mem-002",
            "mem_type": "episodic_memory",
            "content": "用户用 pandas 完成了数据清洗。",
            "source_memory_ids": [],
            "is_important": False,
            "source_session_ids": ["session-115"],
            "created_at": "2026-08-03T14:30:00+08:00"
        }
    ]
}

async def main():
    llm = OpenAICompatibleLLM(
        base_url="https://your-host/v1",
        api_key="your-key",
        model="your-model"
    )
    result = await DreamingExtractor(llm).dream(MemoryBatch.from_dict(payload))
    print(result.to_dict())

asyncio.run(main())
```

如果已有 Agent Memory 的 LLM 对象，可用 `InvokeLLMAdapter(existing_llm)`；其他同步或异步函数可用 `CallableLLM(function)`。真正需要实现的最小接口只有：

```python
class MyLLM:
    async def complete(self, prompt: str) -> str:
        ...
```

## CLI 调用

```bash
cd dreaming-memory
python -m agent_dreaming \
  --input examples/memories.json \
  --output dreamed.json \
  --base-url https://your-host/v1 \
  --api-key your-key \
  --model your-model
```

也可以使用环境变量 `DREAMING_API_BASE`、`DREAMING_API_KEY`、`DREAMING_MODEL`。先检查输入和最终提示词、不调用模型：

```bash
python -m agent_dreaming -i examples/memories.json --print-prompt
```

## 输出格式

```json
{
  "schema_version": "2.0",
  "memories": [
    {
      "memory_id": "dream-<sha256>",
      "mem_type": "user_profile",
      "content": "用户偏好使用 Python 处理数据。",
      "source_memory_ids": ["mem-001", "mem-002"],
      "is_important": false,
      "source_session_ids": ["session-101", "session-115"],
      "created_at": "2026-08-03T14:30:00+08:00"
    }
  ]
}
```

模型返回的 `source_memory_ids` 只能引用本轮实际送入的记忆，模块会严格校验并将已有来源链展开到最终结果，防止无来源结论进入结果。输入超过 `max_input_tokens` 时沿用原 Dreaming “保留首尾、从中间删除”的压缩策略；Python 调用可通过 `result.diagnostics_dict()` 查看 `input_memory_ids` 和 `omitted_memory_ids`，CLI 会把遗漏 ID 写到 stderr。如果仅首尾两条仍超限，调用会明确报错。

## 与原 Agent Memory Dreaming 的边界

| 能力 | 原模块 | 本独立模块 |
|---|---|---|
| 输入 | message store 中的原始会话 | 调用方传入的已萃取记忆 JSON |
| 执行 | 后台周期扫描 | `dream()` 单次调用 |
| checkpoint | KV store | 不需要 |
| 结果存储 | 回写 Agent Memory | 返回结构化结果 |
| LLM | Agent Memory Model | 最小协议 / OpenAI-compatible API / 适配器 |
| 输出溯源 | session ID | 精确到 `source_memory_ids` |

## 验证

```bash
cd dreaming-memory
python -m unittest discover -s tests -v
python -m agent_dreaming -i examples/memories.json --print-prompt
```
