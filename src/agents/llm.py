from __future__ import annotations

import os
from collections.abc import Generator

from openai import OpenAI

from .message import Message, ToolCall


class LLM:
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        self.model = model
        api_key = os.environ.get("OPENAI_API_KEY")
        self._client = OpenAI(api_key=api_key)

    def complete(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        kwargs: dict = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = []
        if choice.message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in choice.message.tool_calls
            ]
        return Message.assistant(content, tool_calls)

    def stream_complete(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> Generator[str, None, Message]:
        kwargs: dict = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = self._client.chat.completions.create(**kwargs)

        content_chunks: list[str] = []
        tool_call_dicts: dict[int, dict] = {}

        for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                content_chunks.append(delta.content)
                yield delta.content

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_dicts:
                        tool_call_dicts[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_call_dicts[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_call_dicts[idx]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_call_dicts[idx]["arguments"] += tc_delta.function.arguments

        tool_calls = [
            ToolCall(
                id=tc["id"],
                name=tc["name"],
                arguments=tc["arguments"],
            )
            for tc in tool_call_dicts.values()
        ]
        return Message.assistant("".join(content_chunks), tool_calls)

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding