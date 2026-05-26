# agents

A small AI agent framework built from scratch to understand how agent systems work internally, instead of relying on frameworks like LangChain. Each module maps to one concept, written so you can read it, understand it, and explain it in an interview.

- **Agent loop** (`agent.py`): send messages and tool schemas to the LLM, execute any tools it requests, feed results back, repeat until a final answer.
- **Tool calling** (`tools.py`): JSON schemas for the model, Python callables for execution, string results fed back into conversation.
- **Context management** (`context.py`): conversation history, trims when over budget, summarizes and compresses longer conversations.
- **Vector store & RAG** (`store.py`): embed text, store vectors in memory, retrieve via cosine similarity — no external infrastructure.

## Project Layout

```
src/agents/
├── message.py    # Message and ToolCall dataclasses
├── llm.py        # OpenAI chat, streaming, and embeddings wrapper
├── store.py      # In-memory vector store with cosine similarity
├── tools.py      # Tool registry and built-in tools
├── context.py    # Conversation history, trimming, summarization, save/load
├── agent.py      # ReAct agent loop and chat interface
└── cli.py        # CLI entry point
```

- `uv sync` to install dependencies
- Add `OPENAI_API_KEY` to a `.env` file, then `uv run agents` to start
- `uv run pytest tests/ -v` to run tests