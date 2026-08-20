# Agent Dreaming（独立记忆二次萃取模块）

本目录把 `agent-memory` 的 Dreaming 核心抽成一个可独立安装、一次调用完成的 Python 包。它接收**已经萃取过的记忆**，通过原 Dreaming 的长期价值筛选、压缩、跨条目归纳、类型约束、重要性判断和失败重试逻辑，输出可追溯的二次萃取结果。

它有意不包含原模块的会话读取、KV checkpoint、向量库存储、后台定时器和遗忘曲线。调用方决定输入来自哪里、结果如何去重与回写。因此本包没有 Agent Memory 运行时依赖，也没有第三方 Python 依赖。

## 输入格式（schema 1.0）

顶层字段：

- `schema_version`：目前固定为 `"1.0"`，可省略。
- `user_id`：必填，非空字符串。
- `scope_id`：可选，默认 `"default"`。
- `batch_id`：可选，调用方的批次追踪 ID。
- `memories`：必填数组；允许空数组。

每条记忆字段：

- `memory_id`：必填，在批次内唯一；也是输出溯源所使用的 ID。
- `mem_type`：必填，只能是 `user_profile`、`semantic_memory`、`episodic_memory`。
- `content`：必填，已经完成第一次萃取的记忆正文。
- `source_session_id`、`created_at`、`metadata`：可选来源信息。
- `is_important`：可选布尔值，默认 `false`。

完整输入见 [examples/memories.json](examples/memories.json)。

## Python 调用

```python
import asyncio
from agent_dreaming import DreamingExtractor, MemoryBatch, OpenAICompatibleLLM

payload = {
    "schema_version": "1.0",
    "user_id": "user-001",
    "scope_id": "assistant-demo",
    "memories": [
        {
            "memory_id": "mem-001",
            "mem_type": "user_profile",
            "content": "用户在项目中优先选择 Python。"
        },
        {
            "memory_id": "mem-002",
            "mem_type": "episodic_memory",
            "content": "用户用 pandas 完成了数据清洗。"
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
  "schema_version": "1.0",
  "memories": [
    {
      "mem_type": "user_profile",
      "content": "用户偏好使用 Python 处理数据。",
      "source_memory_ids": ["mem-001", "mem-002"],
      "is_important": false
    }
  ],
  "input_memory_ids": ["mem-001", "mem-002"],
  "omitted_memory_ids": []
}
```

`source_memory_ids` 只能引用本次实际送入模型的记忆，模块会严格校验，防止无来源结论进入结果。输入超过 `max_input_tokens` 时沿用原 Dreaming “保留首尾、从中间删除”的压缩策略；被压缩掉的 ID 会按原输入顺序出现在 `omitted_memory_ids` 中，不会静默丢失。如果仅首尾两条本身仍超限，调用会明确报错并要求增大限制或缩短输入，不会把超限请求静默发送给模型。

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
