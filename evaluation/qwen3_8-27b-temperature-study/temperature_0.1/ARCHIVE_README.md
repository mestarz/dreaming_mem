# 独立 Dreaming 召回测试存档

- 模型：`qwen3.8:27b`
- 温度：`0.1`
- top_p：`0.1`
- 输入记忆：438 条
- 二次萃取结果：218 条

后续单独召回测试优先使用 `recall_corpus.json`；需要表格工具时使用
`recall_corpus.csv`。`dreaming_result.json` 是独立模块的原生输出，
`input_memories.json` 是本次完整输入，`raw_llm_responses.json` 用于审计模型原文。

运行召回测试前可执行：

```bash
sha256sum -c checksums.sha256
```
