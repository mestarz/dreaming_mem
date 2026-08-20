import unittest

from agent_dreaming import (
    MemoryBatch,
    content_sha256,
    preprocess_memory_batch,
)


def memory(
    memory_id: str,
    mem_type: str,
    content: str,
    created_at: str | None = None,
) -> dict:
    return {
        "memory_id": memory_id,
        "mem_type": mem_type,
        "content": content,
        "source_memory_ids": [],
        "is_important": False,
        "created_at": created_at,
    }


class MemoryPreprocessingTests(unittest.TestCase):
    def batch(self, memories: list[dict]) -> MemoryBatch:
        return MemoryBatch.from_dict({"schema_version": "2.0", "memories": memories})

    def test_core_uses_exact_sha256_and_keeps_newest_timestamp(self):
        source = self.batch(
            [
                memory("old", "user_profile", "用户喜欢王菲", "2026-08-18T10:00:00+08:00"),
                memory("distinct", "user_profile", "用户喜欢王菲。", "2026-08-17T10:00:00+08:00"),
                memory("new", "user_profile", "用户喜欢王菲", "2026-08-20T10:00:00+08:00"),
            ]
        )

        result = preprocess_memory_batch(source)

        self.assertEqual([item.memory_id for item in result.batch.memories], ["distinct", "new"])
        self.assertEqual(result.removed_memory_ids, ("old",))
        self.assertNotEqual(content_sha256("用户喜欢王菲"), content_sha256("用户喜欢王菲。"))

    def test_core_without_time_or_with_tied_time_keeps_later_input(self):
        source = self.batch(
            [
                memory("first", "user_profile", "same"),
                memory("second", "user_profile", "same"),
            ]
        )

        result = preprocess_memory_batch(source)

        self.assertEqual([item.memory_id for item in result.batch.memories], ["second"])
        self.assertEqual(result.removed_memory_ids, ("first",))

    def test_episodic_exact_duplicates_are_preserved(self):
        source = self.batch(
            [
                memory("event-1", "episodic_memory", "用户去了悉尼"),
                memory("event-2", "episodic_memory", "用户去了悉尼"),
            ]
        )

        result = preprocess_memory_batch(source)

        self.assertEqual([item.memory_id for item in result.batch.memories], ["event-1", "event-2"])
        self.assertEqual(result.removed_memory_ids, ())


if __name__ == "__main__":
    unittest.main()
