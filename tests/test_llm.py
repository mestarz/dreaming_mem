import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent_dreaming import CallableLLM, InvokeLLMAdapter, OllamaChatLLM, OpenAICompatibleLLM


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_sync_callable(self):
        client = CallableLLM(lambda prompt: prompt.upper())
        self.assertEqual(await client.complete("hello"), "HELLO")

    async def test_agent_memory_invoke_adapter(self):
        class Fake:
            async def invoke(self, messages):
                return SimpleNamespace(content=messages[0]["content"] + "-ok")

        self.assertEqual(await InvokeLLMAdapter(Fake()).complete("dream"), "dream-ok")

    async def test_openai_compatible_request(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "[]"}}]}).encode()

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            captured["body"] = json.loads(request.data.decode())
            captured["timeout"] = timeout
            return Response()

        client = OpenAICompatibleLLM(
            "https://llm.example/v1",
            "secret",
            "model-x",
            timeout_seconds=12,
            top_p=0.1,
        )
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            # Exercise the HTTP serialization synchronously. ``complete`` is a
            # thin asyncio.to_thread wrapper around this method.
            self.assertEqual(client._complete_blocking("prompt"), "[]")
        self.assertEqual(captured["url"], "https://llm.example/v1/chat/completions")
        self.assertEqual(captured["authorization"], "Bearer secret")
        self.assertEqual(captured["body"]["messages"][0]["content"], "prompt")
        self.assertEqual(captured["body"]["top_p"], 0.1)
        self.assertEqual(captured["timeout"], 12)

    async def test_ollama_native_request_disables_thinking(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return json.dumps({"message": {"role": "assistant", "content": "[]"}}).encode()

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode())
            captured["timeout"] = timeout
            return Response()

        client = OllamaChatLLM(
            "http://127.0.0.1:11434",
            "qwen3.8:27b",
            timeout_seconds=30,
            temperature=0.2,
            top_p=0.1,
            think=False,
            num_ctx=131_072,
            num_predict=65_536,
            format_schema={"type": "array", "maxItems": 100},
        )
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            self.assertEqual(client._complete_blocking("prompt"), "[]")
        self.assertEqual(captured["url"], "http://127.0.0.1:11434/api/chat")
        self.assertFalse(captured["body"]["think"])
        self.assertEqual(captured["body"]["options"]["temperature"], 0.2)
        self.assertEqual(captured["body"]["options"]["top_p"], 0.1)
        self.assertEqual(captured["body"]["options"]["num_ctx"], 131_072)
        self.assertEqual(captured["body"]["options"]["num_predict"], 65_536)
        self.assertEqual(captured["body"]["format"]["maxItems"], 100)
        self.assertEqual(captured["timeout"], 30)


if __name__ == "__main__":
    unittest.main()
