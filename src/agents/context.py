from __future__ import annotations

import json
from pathlib import Path

from .message import Message, ToolCall


def _estimate_tokens(messages: list[Message]) -> int:
    total = 0
    for m in messages:
        total += len(m.content) // 4
        for tc in m.tool_calls:
            total += len(tc.arguments) // 4
            total += len(tc.name) // 4
    return total


class Context:
    def __init__(self, system_prompt: str = "You are a helpful assistant.", max_tokens: int = 4096) -> None:
        self.messages: list[Message] = [Message.system(system_prompt)]
        self.max_tokens = max_tokens

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def get_messages(self) -> list[Message]:
        return list(self.messages)

    def token_count(self) -> int:
        return _estimate_tokens(self.messages)

    def trim(self) -> None:
        if self.token_count() <= self.max_tokens:
            return
        system = self.messages[0]
        conversation = self.messages[1:]
        while len(conversation) > 1 and _estimate_tokens([system] + conversation) > self.max_tokens:
            conversation = conversation[1:]
        self.messages = [system] + conversation

    def summarize(self, llm) -> str:
        conversation = self.messages[1:]
        if not conversation:
            return ""
        text_parts = []
        for m in conversation:
            if m.role == "user":
                text_parts.append(f"User: {m.content}")
            elif m.role == "assistant":
                text_parts.append(f"Assistant: {m.content}")
            elif m.role == "tool":
                text_parts.append(f"Tool ({m.name}): {m.content}")
        conversation_text = "\n".join(text_parts)
        summary_prompt = f"Summarize this conversation briefly, preserving key facts and decisions:\n\n{conversation_text}"
        summary_ctx = Context(system_prompt="You summarize conversations concisely.")
        summary_ctx.add(Message.user(summary_prompt))
        response = llm.complete(summary_ctx.get_messages())
        return response.content

    def summarize_and_compress(self, llm) -> None:
        if self.token_count() <= self.max_tokens:
            return
        system = self.messages[0]
        summary = self.summarize(llm)
        summary_message = Message.system(
            f"{system.content}\n\nSummary of earlier conversation:\n{summary}"
        )
        self.messages = [summary_message]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [m.to_dict() for m in self.messages]
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Context:
        path = Path(path)
        data = json.loads(path.read_text())
        ctx = cls()
        ctx.messages = []
        for d in data:
            tool_calls_parsed = [
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                )
                for tc in d.get("tool_calls", [])
            ]
            m = Message(
                role=d["role"],
                content=d.get("content", ""),
                tool_calls=tool_calls_parsed,
                tool_call_id=d.get("tool_call_id"),
                name=d.get("name"),
            )
            ctx.messages.append(m)
        return ctx

