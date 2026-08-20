# 预检运行（不用于召回测试）

这里保留加入 Ollama 服务端 JSON Schema 之前的两次诊断运行，作为为什么必须
使用解码层结构约束的证据，不能作为正式温度对照或召回语料。

- `pre_schema_temperature_0.1_overlimit`：模型忽略提示词里的 100 条上限，原始
  响应包含 165 个对象；独立模块按上限保留了前 100 条。该目录保留了当时的
  recall corpus，但它只用于诊断，正式召回请勿使用。
- `pre_schema_temperature_0.2_truncated`：模型已生成 197 个对象时响应被截断，
  随后的格式重试返回空数组，因此目录中的 0 条结果不是有效二次萃取结果。

正式结果位于同级 `temperature_0.1`、`temperature_0.2`、`temperature_0.5`、
`temperature_1` 目录，均使用相同 `num_predict`、上下文长度与输入数据。正式运行
不使用 Ollama 的 format/schema 捷径：模型自然完成数组，独立模块统一做 JSON、
字段、类型和来源校验，并保留前 100 条。
