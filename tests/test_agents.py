from agents.agent import Agent
from agents.context import Context
from agents.message import Message, ToolCall
from agents.store import VectorStore
from agents.tools import ToolRegistry, create_default_tools


class TestMessage:
    def test_system_message(self):
        m = Message.system("hello")
        assert m.role == "system"
        assert m.content == "hello"

    def test_user_message(self):
        m = Message.user("hi")
        assert m.role == "user"
        assert m.content == "hi"

    def test_assistant_message(self):
        m = Message.assistant("response")
        assert m.role == "assistant"
        assert m.content == "response"

    def test_tool_result_message(self):
        m = Message.tool_result("call_1", "calculate", "42")
        assert m.role == "tool"
        assert m.tool_call_id == "call_1"
        assert m.name == "calculate"

    def test_to_dict_basic(self):
        m = Message.user("hello")
        d = m.to_dict()
        assert d == {"role": "user", "content": "hello"}

    def test_to_dict_with_tool_calls(self):
        tc = ToolCall(id="call_1", name="calculate", arguments='{"expression": "2+2"}')
        m = Message.assistant("", [tc])
        d = m.to_dict()
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["function"]["name"] == "calculate"

    def test_to_dict_tool_result(self):
        m = Message.tool_result("call_1", "calculate", "42")
        d = m.to_dict()
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "call_1"
        assert d["name"] == "calculate"


class TestToolRegistry:
    def test_register_and_execute(self):
        registry = ToolRegistry()
        registry.register(
            name="echo",
            description="Echoes input",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            fn=lambda text: text,
        )
        result = registry.execute("echo", '{"text": "hello"}')
        assert result == "hello"

    def test_unknown_tool(self):
        registry = ToolRegistry()
        assert "Error" in registry.execute("nonexistent", "{}")

    def test_get_schemas(self):
        registry = create_default_tools()
        schemas = registry.get_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "calculate" in names
        assert "get_weather" in names

    def test_calculate(self):
        registry = create_default_tools()
        result = registry.execute("calculate", '{"expression": "2 + 2"}')
        assert result == "4"

    def test_calculate_rejects_invalid(self):
        registry = create_default_tools()
        result = registry.execute("calculate", '{"expression": "__import__(\\"os\\")"}')
        assert "Error" in result

    def test_get_weather(self):
        registry = create_default_tools()
        result = registry.execute("get_weather", '{"city": "Tokyo"}')
        assert "22" in result

    def test_get_weather_unknown_city(self):
        registry = create_default_tools()
        result = registry.execute("get_weather", '{"city": "Atlantis"}')
        assert "No weather data" in result

    def test_search_knowledge_not_registered_without_store(self):
        registry = create_default_tools()
        schemas = registry.get_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "search_knowledge" not in names

    def test_search_knowledge_registered_with_store(self):
        store = VectorStore()
        mock_llm = object()
        registry = create_default_tools(store=store, llm=mock_llm)
        schemas = registry.get_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "search_knowledge" in names

    def test_search_knowledge_executes_vector_search(self):
        class FakeLLM:
            def embed(self, text):
                return [1.0, 0.0] if "python" in text.lower() else [0.0, 1.0]

        store = VectorStore()
        store.add("Python uses indentation.", [1.0, 0.0], {"source": "python-notes"})
        store.add("Rust has ownership.", [0.0, 1.0], {"source": "rust-notes"})
        registry = create_default_tools(store=store, llm=FakeLLM())

        result = registry.execute("search_knowledge", '{"query": "python syntax"}')

        assert "python-notes" in result
        assert "Python uses indentation." in result


class TestVectorStore:
    def test_add_and_query(self):
        store = VectorStore()
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        store.add("hello world", emb1, {"source": "test"})
        store.add("foo bar", emb2, {"source": "test2"})

        results = store.query([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "hello world"
        assert results[0]["metadata"]["source"] == "test"

    def test_empty_store(self):
        store = VectorStore()
        results = store.query([1.0, 0.0], top_k=3)
        assert results == []

    def test_top_k(self):
        store = VectorStore()
        store.add("a", [1.0, 0.0], {})
        store.add("b", [0.9, 0.1], {})
        store.add("c", [0.0, 1.0], {})
        results = store.query([1.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0]["text"] == "a"

    def test_add_documents(self):
        store = VectorStore()

        def embed_fn(text):
            return [1.0, 0.0] if text == "doc one" else [0.0, 1.0]

        store.add_documents(["doc one", "doc two"], embed_fn=embed_fn, metadata={"source": "test"})

        assert len(store._texts) == 2
        assert store._texts[0] == "doc one"

    def test_add_documents_uses_single_text_embed_fn(self):
        store = VectorStore()
        calls = []

        def embed_fn(text):
            calls.append(text)
            return [1.0, 0.0] if text == "doc one" else [0.0, 1.0]

        store.add_documents(["doc one", "doc two"], embed_fn=embed_fn)

        assert calls == ["doc one", "doc two"]
        assert store.query([1.0, 0.0], top_k=1)[0]["text"] == "doc one"

    def test_search_with_embed_fn(self):
        store = VectorStore()
        store.add("python is great", [1.0, 0.0])
        store.add("rust is fast", [0.0, 1.0])
        calls = []

        def embed_fn(text):
            calls.append(text)
            return [1.0, 0.0] if "python" in text.lower() else [0.0, 1.0]

        results = store.search("python programming", embed_fn=embed_fn, top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "python is great"


class TestContext:
    def test_add_and_get_messages(self):
        ctx = Context()
        ctx.add(Message.user("hello"))
        messages = ctx.get_messages()
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].content == "hello"

    def test_token_count(self):
        ctx = Context()
        ctx.add(Message.user("a" * 100))
        count = ctx.token_count()
        assert count > 0

    def test_trim_no_op_when_under_limit(self):
        ctx = Context(max_tokens=10000)
        ctx.add(Message.user("short message"))
        before = len(ctx.messages)
        ctx.trim()
        assert len(ctx.messages) == before

    def test_trim_removes_old_messages(self):
        ctx = Context(max_tokens=10)
        for i in range(20):
            ctx.add(Message.user(f"message number {i} with enough text to consume tokens"))
        assert ctx.token_count() > 10
        ctx.trim()
        assert ctx.token_count() <= 10 or len(ctx.messages) <= 3

    def test_trim_preserves_system_prompt(self):
        ctx = Context(system_prompt="important system prompt", max_tokens=10)
        ctx.add(Message.user("a" * 500))
        ctx.add(Message.user("b" * 500))
        ctx.trim()
        assert ctx.messages[0].role == "system"
        assert ctx.messages[0].content == "important system prompt"

    def test_trim_keeps_as_many_recent_messages_as_fit(self):
        ctx = Context(system_prompt="", max_tokens=9)
        for _ in range(4):
            ctx.add(Message.user("a" * 12))
        ctx.trim()
        assert len(ctx.messages) == 4

    def test_save_and_load(self, tmp_path):
        ctx = Context(system_prompt="test system")
        ctx.add(Message.user("hello"))
        ctx.add(Message.assistant("hi there"))

        path = tmp_path / "context.json"
        ctx.save(path)

        loaded = Context.load(path)
        assert len(loaded.messages) == 3
        assert loaded.messages[0].content == "test system"
        assert loaded.messages[1].content == "hello"
        assert loaded.messages[2].content == "hi there"

    def test_save_and_load_with_tool_calls(self, tmp_path):
        ctx = Context()
        tc = ToolCall(id="call_1", name="calculate", arguments='{"expression": "2+2"}')
        ctx.add(Message.user("what is 2+2?"))
        ctx.add(Message.assistant("", [tc]))
        ctx.add(Message.tool_result("call_1", "calculate", "4"))

        path = tmp_path / "context_tool.json"
        ctx.save(path)

        loaded = Context.load(path)
        assert len(loaded.messages) == 4
        assert loaded.messages[2].tool_calls[0].name == "calculate"
        assert loaded.messages[3].tool_call_id == "call_1"

    def test_summarize_uses_llm(self):
        class FakeLLM:
            def complete(self, messages):
                assert "User: remember this" in messages[-1].content
                return Message.assistant("User asked to remember something.")

        ctx = Context()
        ctx.add(Message.user("remember this"))

        assert ctx.summarize(FakeLLM()) == "User asked to remember something."

    def test_summarize_and_compress_replaces_history_with_summary(self):
        class FakeLLM:
            def complete(self, messages):
                return Message.assistant("Compressed summary.")

        ctx = Context(system_prompt="base instructions", max_tokens=5)
        ctx.add(Message.user("a" * 100))

        ctx.summarize_and_compress(FakeLLM())

        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == "system"
        assert "base instructions" in ctx.messages[0].content
        assert "Compressed summary." in ctx.messages[0].content


class TestAgent:
    def test_run_returns_final_response(self):
        class FakeLLM:
            def complete(self, messages, tools=None):
                return Message.assistant("hello")

        agent = Agent(llm=FakeLLM(), context=Context(), tools=ToolRegistry())

        assert agent.run("hi") == "hello"

    def test_run_executes_tool_calls(self):
        class FakeLLM:
            def __init__(self):
                self.calls = 0

            def complete(self, messages, tools=None):
                self.calls += 1
                if self.calls == 1:
                    return Message.assistant(
                        "",
                        [ToolCall(id="call_1", name="echo", arguments='{"text": "hello"}')],
                    )
                return Message.assistant(f"tool said {messages[-1].content}")

        registry = ToolRegistry()
        registry.register(
            name="echo",
            description="Echo text",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            fn=lambda text: text,
        )
        agent = Agent(llm=FakeLLM(), context=Context(), tools=registry)

        assert agent.run("use a tool") == "tool said hello"

    def test_run_streaming_captures_generator_return(self, capsys):
        class FakeLLM:
            def stream_complete(self, messages, tools=None):
                yield "hel"
                yield "lo"
                return Message.assistant("hello")

        agent = Agent(llm=FakeLLM(), context=Context(), tools=ToolRegistry())

        assert agent.run_streaming("hi") == "hello"
        assert "hello" in capsys.readouterr().out
