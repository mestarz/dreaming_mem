import tempfile
import unittest
from pathlib import Path

from agent_dreaming import (
    MemoryBatch,
    SchemaError,
    memory_batch_from_csv,
    memory_batch_to_csv_text,
)


class CompactCsvTests(unittest.TestCase):
    def write_csv(self, text: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "memory.csv"
        path.write_text(text, encoding="utf-8-sig")
        return path

    def test_loads_compact_csv_and_maps_source_types(self):
        path = self.write_csv(
            "id,type,topic,subtopic,content\n"
            "1,core,兴趣爱好,音乐偏好,用户喜欢王菲\n"
            "2,episodic,旅行,澳洲,用户去过悉尼\n"
            "3,semantic,知识,方法,悉尼位于澳大利亚\n"
        )
        batch = memory_batch_from_csv(path)
        self.assertEqual([item.memory_id for item in batch.memories], ["1", "2", "3"])
        self.assertEqual(
            [item.mem_type.value for item in batch.memories],
            ["user_profile", "episodic_memory", "semantic_memory"],
        )
        self.assertEqual(batch.memories[0].content, "用户喜欢王菲")
        self.assertEqual(batch.memories[0].topic, "兴趣爱好")
        self.assertEqual(batch.memories[0].subtopic, "音乐偏好")
        self.assertEqual(batch.memories[0].source_memory_ids, ())

    def test_timestamped_csv_deduplicates_only_core_and_keeps_newest(self):
        path = self.write_csv(
            "id,type,topic,subtopic,content,created_at,updated_at\n"
            "core-new,core,偏好,音乐,用户喜欢王菲,2026-08-20 09:00:00,2026-08-20 11:00:00\n"
            "event-1,episodic,旅行,悉尼,用户去了悉尼,2026-08-19 09:00:00,2026-08-19 09:00:00\n"
            "core-old,core,偏好,音乐,用户喜欢王菲,2026-08-18 09:00:00,2026-08-18 11:00:00\n"
            "event-2,episodic,旅行,悉尼,用户去了悉尼,2026-08-19 09:00:00,2026-08-19 09:00:00\n"
        )

        batch = memory_batch_from_csv(path)

        self.assertEqual(
            [item.memory_id for item in batch.memories],
            ["core-new", "event-1", "event-2"],
        )

    def test_five_column_core_duplicates_keep_later_row(self):
        path = self.write_csv(
            "id,type,topic,subtopic,content\n"
            "old,core,偏好,音乐,用户喜欢王菲\n"
            "new,core,偏好,音乐,用户喜欢王菲\n"
        )

        batch = memory_batch_from_csv(path)

        self.assertEqual([item.memory_id for item in batch.memories], ["new"])

    def test_equal_updated_at_uses_created_at_before_row_order(self):
        path = self.write_csv(
            "id,type,topic,subtopic,content,created_at,updated_at\n"
            "new,core,偏好,音乐,用户喜欢王菲,2026-08-20 09:00:00,2026-08-21 09:00:00\n"
            "old,core,偏好,音乐,用户喜欢王菲,2026-08-18 09:00:00,2026-08-21 09:00:00\n"
        )

        batch = memory_batch_from_csv(path)

        self.assertEqual([item.memory_id for item in batch.memories], ["new"])

    def test_drops_orphan_subtopic_without_inventing_topic(self):
        path = self.write_csv(
            "id,type,topic,subtopic,content\n"
            "1,core,,音乐偏好,用户喜欢王菲\n"
        )

        batch = memory_batch_from_csv(path)

        self.assertIsNone(batch.memories[0].topic)
        self.assertIsNone(batch.memories[0].subtopic)

    def test_round_trips_csv_quoting_and_newlines(self):
        batch = MemoryBatch.from_dict(
            {
                "schema_version": "2.0",
                "memories": [
                    {
                        "memory_id": "1",
                        "mem_type": "user_profile",
                        "topic": "偏好,习惯",
                        "subtopic": "音乐",
                        "content": "用户喜欢王菲。\n也喜欢林俊杰。",
                        "source_memory_ids": [],
                        "is_important": False,
                    }
                ],
            }
        )
        path = self.write_csv(memory_batch_to_csv_text(batch))
        self.assertEqual(memory_batch_from_csv(path), batch)

    def test_empty_batch_still_writes_header(self):
        self.assertEqual(
            memory_batch_to_csv_text(MemoryBatch(())),
            "id,type,topic,subtopic,content\n",
        )

    def test_requires_exact_columns(self):
        path = self.write_csv("id,type,content\n1,core,x\n")
        with self.assertRaisesRegex(SchemaError, "columns must be exactly"):
            memory_batch_from_csv(path)

    def test_rejects_unknown_type(self):
        path = self.write_csv("id,type,topic,subtopic,content\n1,unknown,,,x\n")
        with self.assertRaisesRegex(SchemaError, "unsupported type"):
            memory_batch_from_csv(path)

    def test_rejects_duplicate_or_empty_ids_through_canonical_validation(self):
        duplicate = self.write_csv(
            "id,type,topic,subtopic,content\n1,core,,,x\n1,core,,,y\n"
        )
        with self.assertRaisesRegex(SchemaError, "unique"):
            memory_batch_from_csv(duplicate)
        empty = self.write_csv("id,type,topic,subtopic,content\n,core,,,x\n")
        with self.assertRaisesRegex(SchemaError, "memory_id"):
            memory_batch_from_csv(empty)


if __name__ == "__main__":
    unittest.main()
