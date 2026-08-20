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
            "schema_version": "2.0",
            "memories": memories
            if memories is not None
            else [
                {
                    "memory_id": "m1",
                    "mem_type": "user_profile",
                    "content": "用户偏好 Python。",
                    "source_memory_ids": [],
                    "is_important": False,
                },
                {
                    "memory_id": "m2",
                    "mem_type": "episodic_memory",
                    "content": "用户用 pandas 完成了清洗。",
                    "source_memory_ids": [],
                    "is_important": False,
                },
            ],
        }
    )


class SchemaTests(unittest.TestCase):
    def test_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(SchemaError, "unique"):
            batch(
                [
                    {"memory_id": "same", "mem_type": "user_profile", "content": "a", "source_memory_ids": [], "is_important": False},
                    {"memory_id": "same", "mem_type": "semantic_memory", "content": "b", "source_memory_ids": [], "is_important": False},
                ]
            )

    def test_rejects_unknown_type(self):
        with self.assertRaisesRegex(SchemaError, "must be one of"):
            batch([{"memory_id": "m1", "mem_type": "summary", "content": "x", "source_memory_ids": [], "is_important": False}])

    def test_rejects_unknown_fields(self):
        with self.assertRaisesRegex(SchemaError, "unknown fields: metadata"):
            batch([{"memory_id": "m1", "mem_type": "semantic_memory", "content": "x", "source_memory_ids": [], "is_important": False, "metadata": {}}])

    def test_rejects_removed_batch_fields(self):
        with self.assertRaisesRegex(SchemaError, "unknown fields: user_id"):
            MemoryBatch.from_dict({"schema_version": "2.0", "user_id": "u1", "memories": []})

    def test_rejects_schema_v1(self):
        with self.assertRaisesRegex(SchemaError, "expected '2.0'"):
            MemoryBatch.from_dict({"schema_version": "1.0", "memories": []})

    def test_round_trip(self):
        original = batch()
        self.assertEqual(MemoryBatch.from_dict(original.to_dict()), original)

    def test_rejects_timestamp_without_timezone(self):
        with self.assertRaisesRegex(SchemaError, "timezone offset"):
            batch(
                [
                    {
                        "memory_id": "m1",
                        "mem_type": "semantic_memory",
                        "content": "x",
                        "source_memory_ids": [],
                        "is_important": False,
                        "created_at": "2026-08-20T10:00:00",
                    }
                ]
            )

    def test_rejects_self_referential_lineage(self):
        with self.assertRaisesRegex(SchemaError, "must not reference the memory itself"):
            batch(
                [
                    {
                        "memory_id": "m1",
                        "mem_type": "semantic_memory",
                        "content": "x",
                        "source_memory_ids": ["m1"],
                        "is_important": False,
                    }
                ]
            )


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
        self.assertTrue(result.memories[0].memory_id.startswith("dream-"))
        self.assertEqual(set(result.memories[0].to_dict()), set(batch().memories[0].to_dict()))
        self.assertEqual(result.memories[0].source_memory_ids, ("m1", "m2"))
        self.assertEqual(result.input_memory_ids, ("m1", "m2"))
        self.assertEqual(result.omitted_memory_ids, ())
        self.assertEqual(MemoryBatch.from_dict(result.to_dict()).memories, result.memories)

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

    async def test_rejects_non_boolean_output_flag(self):
        async def respond(_):
            return json.dumps(
                [
                    {
                        "mem_type": "user_profile",
                        "content": "unsupported",
                        "source_memory_ids": ["m1"],
                        "is_important": "false",
                    }
                ]
            )

        extractor = DreamingExtractor(CallableLLM(respond), DreamingConfig(retries=1))
        with self.assertRaisesRegex(SchemaError, "non-boolean is_important"):
            await extractor.dream(batch())

    async def test_output_propagates_source_context(self):
        source_batch = batch(
            [
                {
                    "memory_id": "m1",
                    "mem_type": "episodic_memory",
                    "content": "older",
                    "source_memory_ids": [],
                    "is_important": False,
                    "source_session_ids": ["s2"],
                    "created_at": "2026-08-19T09:00:00+08:00",
                },
                {
                    "memory_id": "m2",
                    "mem_type": "episodic_memory",
                    "content": "newer",
                    "source_memory_ids": [],
                    "is_important": False,
                    "source_session_ids": ["s1"],
                    "created_at": "2026-08-20T10:00:00+08:00",
                },
            ]
        )

        async def respond(_):
            return json.dumps(
                [
                    {
                        "mem_type": "episodic_memory",
                        "content": "merged",
                        "source_memory_ids": ["m1", "m2"],
                        "is_important": False,
                    }
                ]
            )

        result = await DreamingExtractor(CallableLLM(respond)).dream(source_batch)
        self.assertEqual(result.memories[0].source_session_ids, ("s1", "s2"))
        self.assertEqual(result.memories[0].created_at, "2026-08-20T10:00:00+08:00")

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

    async def test_output_ids_are_stable_and_lineage_survives_another_round(self):
        async def first_response(_):
            return json.dumps(
                [{"mem_type": "semantic_memory", "content": "stable", "source_memory_ids": ["m1", "m2"], "is_important": False}]
            )

        first = await DreamingExtractor(CallableLLM(first_response)).dream(batch())
        first_id = first.memories[0].memory_id

        async def second_response(_):
            return json.dumps(
                [{"mem_type": "semantic_memory", "content": "stable", "source_memory_ids": [first_id], "is_important": False}]
            )

        second = await DreamingExtractor(CallableLLM(second_response)).dream(first.to_dict())
        self.assertEqual(second.memories[0].memory_id, first_id)
        self.assertEqual(second.memories[0].source_memory_ids, ("m1", "m2"))

    async def test_middle_drop_is_visible_in_result(self):
        memories = [
            {"memory_id": f"m{i}", "mem_type": "semantic_memory", "content": "x" * 120, "source_memory_ids": [], "is_important": False}
            for i in range(6)
        ]

        async def respond(_):
            return "[]"

        extractor = DreamingExtractor(CallableLLM(respond), DreamingConfig(max_input_tokens=140))
        result = await extractor.dream(batch(memories))
        self.assertEqual(result.input_memory_ids, ("m0", "m5"))
        self.assertEqual(result.omitted_memory_ids, ("m1", "m2", "m3", "m4"))

    async def test_oversized_retained_pair_fails_instead_of_silently_exceeding_limit(self):
        memories = [
            {"memory_id": f"m{i}", "mem_type": "semantic_memory", "content": "x" * 500, "source_memory_ids": [], "is_important": False}
            for i in range(2)
        ]
        extractor = DreamingExtractor(CallableLLM(lambda _: "[]"), DreamingConfig(max_input_tokens=20))
        with self.assertRaisesRegex(SchemaError, "exceed max_input_tokens"):
            await extractor.dream(batch(memories))

    async def test_dream_accepts_raw_mapping(self):
        result = await DreamingExtractor(CallableLLM(lambda _: "[]")).dream(
            {"schema_version": "2.0", "memories": []}
        )
        self.assertEqual(result.to_dict()["memories"], [])


if __name__ == "__main__":
    unittest.main()
