from __future__ import annotations

import json
from typing import Callable

from .llm import LLM
from .store import VectorStore


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, dict] = {}
        self._functions: dict[str, Callable] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        fn: Callable,
    ) -> None:
        self._tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        self._functions[name] = fn

    def get_schemas(self) -> list[dict]:
        return list(self._tools.values())

    def execute(self, name: str, arguments: str) -> str:
        fn = self._functions.get(name)
        if fn is None:
            return f"Error: unknown tool '{name}'"
        try:
            args = json.loads(arguments)
            result = fn(**args)
            return str(result)
        except Exception as e:
            return f"Error: {e}"


def _calculate(expression: str) -> str:
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters in expression"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def _get_weather(city: str) -> str:
    weather_data = {
        "tokyo": "22°C, partly cloudy",
        "london": "14°C, rainy",
        "paris": "18°C, overcast",
        "new york": "25°C, sunny",
        "sydney": "19°C, clear",
        "berlin": "12°C, windy",
        "madrid": "28°C, hot and sunny",
        "moscow": "5°C, snowing",
        "beijing": "20°C, hazy",
        "mumbai": "33°C, humid",
    }
    key = city.lower().strip()
    return weather_data.get(key, f"No weather data for {city}. Available: {', '.join(weather_data.keys())}")


def create_default_tools(
    store: VectorStore | None = None,
    llm: LLM | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        name="calculate",
        description="Evaluate a math expression. Supports basic arithmetic.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"},
            },
            "required": ["expression"],
        },
        fn=_calculate,
    )
    registry.register(
        name="get_weather",
        description="Get current weather for a city.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
        fn=_get_weather,
    )
    if store is not None and llm is not None:
        def _search_knowledge(query: str) -> str:
            results = store.search(query, embed_fn=llm.embed, top_k=3)
            if not results:
                return "No documents found in the knowledge base."
            return "\n\n".join(
                f"[{r['metadata'].get('source', 'unknown')}] (score: {r['score']:.2f})\n{r['text']}"
                for r in results
            )

        registry.register(
            name="search_knowledge",
            description="Search the knowledge base for relevant information. Use this when you need facts or context from documents.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
            fn=_search_knowledge,
        )
    return registry