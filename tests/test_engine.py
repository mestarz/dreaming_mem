import json
import unittest

from agent_dreaming import (
    CallableLLM,
    DreamingConfig,
    DreamingExtractor,
    MemoryBatch,
    SchemaError,
)


def batch(memories=None):
    return MemoryBatch.from_dict(
        {
            "schema_version": "1.0",
            "user_id": "u1",
            "scope_id": "s1",
            "memories": memories
            if memories is not None
            else [
                {"memory_id": "m1", "mem_type": "user_profile", "content": "用户偏好 Python。"},
                {"memory_id": "m2", "mem_type": "episodic_memory", "content": "用户用 pandas 完成了清洗。"},
            ],
        }
    )


class SchemaTests(unittest.TestCase):
    def test_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(SchemaError, "unique"):
            batch(
                [
                    {"memory_id": "same", "mem_type": "user_profile", "content": "a"},
                    {"memory_id": "same", "mem_type": "semantic_memory", "content": "b"},
                ]
            )

    def test_rejects_unknown_type(self):
        with self.assertRaisesRegex(SchemaError, "must be one of"):
            batch([{"memory_id": "m1", "mem_type": "summary", "content": "x"}])

    def test_round_trip(self):
        original = batch()
        self.assertEqual(MemoryBatch.from_dict(original.to_dict()), original)


class EngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_second_stage_extraction(self):
        async def respond(prompt):
            self.assertIn('"memory_id":"m1"', prompt)
            return json.dumps(
                [
                    {
                        "mem_type": "user_profile",
                        "content": "用户偏好使用 Python 处理数据。",
                        "source_memory_ids": ["m1", "m2"],
                        "is_important": False,
                    }
                ],
                ensure_ascii=False,
            )

        result = await DreamingExtractor(CallableLLM(respond)).dream(batch())
        self.assertEqual(len(result.memories), 1)
        self.assertEqual(result.memories[0].source_memory_ids, ("m1", "m2"))
        self.assertEqual(result.input_memory_ids, ("m1", "m2"))
        self.assertEqual(result.omitted_memory_ids, ())

    async def test_empty_input_skips_llm(self):
        called = False

        async def respond(_):
            nonlocal called
            called = True
            return "[]"

        result = await DreamingExtractor(CallableLLM(respond)).dream(batch([]))
        self.assertFalse(called)
        self.assertEqual(result.memories, ())

    async def test_retries_bad_json_with_validation_feedback(self):
        prompts = []

        async def respond(prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                return "not json"
            return "[]"

        result = await DreamingExtractor(CallableLLM(respond)).dream(batch())
        self.assertEqual(result.memories, ())
        self.assertEqual(len(prompts), 2)
        self.assertIn("上一次输出未通过格式校验", prompts[1])

    async def test_rejects_untraceable_output(self):
        async def respond(_):
            return json.dumps(
                [
                    {
                        "mem_type": "user_profile",
                        "content": "unsupported",
                        "source_memory_ids": ["does-not-exist"],
                        "is_important": False,
                    }
                ]
            )

        extractor = DreamingExtractor(CallableLLM(respond), DreamingConfig(retries=1))
        with self.assertRaisesRegex(SchemaError, "unknown source_memory_id"):
            await extractor.dream(batch())

    async def test_caps_and_merges_exact_duplicates(self):
        async def respond(_):
            return json.dumps(
                [
                    {
                        "mem_type": "semantic_memory",
                        "content": "same",
                        "source_memory_ids": ["m1"],
                        "is_important": False,
                    },
                    {
                        "mem_type": "semantic_memory",
                        "content": "same",
                        "source_memory_ids": ["m2"],
                        "is_important": True,
                    },
                    {
                        "mem_type": "semantic_memory",
                        "content": "capped",
                        "source_memory_ids": ["m1"],
                        "is_important": False,
                    },
                ]
            )

        extractor = DreamingExtractor(CallableLLM(respond), DreamingConfig(max_output_items=2))
        result = await extractor.dream(batch())
        self.assertEqual(len(result.memories), 1)
        self.assertEqual(result.memories[0].source_memory_ids, ("m1", "m2"))
        self.assertTrue(result.memories[0].is_important)

    async def test_middle_drop_is_visible_in_result(self):
        memories = [
            {"memory_id": f"m{i}", "mem_type": "semantic_memory", "content": "x" * 120}
            for i in range(6)
        ]

        async def respond(_):
            return "[]"

        extractor = DreamingExtractor(CallableLLM(respond), DreamingConfig(max_input_tokens=120))
        result = await extractor.dream(batch(memories))
        self.assertEqual(result.input_memory_ids, ("m0", "m5"))
        self.assertEqual(result.omitted_memory_ids, ("m1", "m2", "m3", "m4"))

    async def test_oversized_retained_pair_fails_instead_of_silently_exceeding_limit(self):
        memories = [
            {"memory_id": f"m{i}", "mem_type": "semantic_memory", "content": "x" * 500}
            for i in range(2)
        ]
        extractor = DreamingExtractor(CallableLLM(lambda _: "[]"), DreamingConfig(max_input_tokens=20))
        with self.assertRaisesRegex(SchemaError, "exceed max_input_tokens"):
            await extractor.dream(batch(memories))

    async def test_dream_accepts_raw_mapping(self):
        result = await DreamingExtractor(CallableLLM(lambda _: "[]")).dream(
            {"user_id": "u1", "memories": []}
        )
        self.assertEqual(result.to_dict()["memories"], [])


if __name__ == "__main__":
    unittest.main()
