# AGENTS.md

Small AI agent framework built from scratch for learning. Prefer readable code over clever code. Keep each module about one idea. Do not add frameworks that hide the learning value.

The project covers a ReAct-style agent loop, structured tool calling with JSON schemas, conversation context management with trimming and summarization, embeddings with an in-memory vector store using cosine similarity, and a basic RAG workflow. Stay focused on those fundamentals.

Do not add LangChain, external vector databases, multi-agent orchestration, MCP, workflow engines, SQLite persistence, or extra provider abstractions.

Snake_case for files and functions, PascalCase for classes. Internal imports stay package-relative. Tools return error strings, core logic raises exceptions. Use single tests while iterating, run the full suite before calling work done.