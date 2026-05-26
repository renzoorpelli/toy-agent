from __future__ import annotations

from .context import Context
from .llm import LLM
from .message import Message
from .tools import ToolRegistry


class Agent:
    def __init__(
        self,
        llm: LLM,
        context: Context,
        tools: ToolRegistry | None = None,
        max_iterations: int = 10,
    ) -> None:
        self.llm = llm
        self.context = context
        self.tools = tools or ToolRegistry()
        self.max_iterations = max_iterations

    def run(self, user_input: str, verbose: bool = False) -> str:
        self.context.add(Message.user(user_input))

        for _ in range(self.max_iterations):
            messages = self.context.get_messages()
            tool_schemas = self.tools.get_schemas() or None
            response = self.llm.complete(messages, tools=tool_schemas)

            if verbose and response.content:
                print(f"\n[thinking] {response.content}")

            if not response.tool_calls:
                self.context.add(response)
                return response.content

            self.context.add(response)

            for tc in response.tool_calls:
                if verbose:
                    print(f"\n[tool call] {tc.name}({tc.arguments})")
                result = self.tools.execute(tc.name, tc.arguments)
                if verbose:
                    print(f"[tool result] {result}")
                self.context.add(Message.tool_result(tc.id, tc.name, result))

        return "I couldn't complete this task within the iteration limit."

    def run_streaming(self, user_input: str, verbose: bool = False) -> str:
        self.context.add(Message.user(user_input))

        for _ in range(self.max_iterations):
            messages = self.context.get_messages()
            tool_schemas = self.tools.get_schemas() or None

            gen = self.llm.stream_complete(messages, tools=tool_schemas)

            final_message: Message | None = None

            while True:
                try:
                    token = next(gen)
                except StopIteration as stop:
                    final_message = stop.value
                    break
                print(token, end="", flush=True)

            if final_message is None:
                final_message = Message.assistant("")

            if not final_message.tool_calls:
                self.context.add(final_message)
                print()
                return final_message.content

            self.context.add(final_message)

            for tc in final_message.tool_calls:
                if verbose:
                    print(f"\n[tool call] {tc.name}({tc.arguments})")
                result = self.tools.execute(tc.name, tc.arguments)
                if verbose:
                    print(f"[tool result] {result}")
                self.context.add(Message.tool_result(tc.id, tc.name, result))

            print("\n[continuing...]\n")

        return "I couldn't complete this task within the iteration limit."

    def chat(self, verbose: bool = False, stream: bool = True) -> None:
        print("Agent ready. Type 'quit' to exit.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Bye!")
                break

            if stream:
                run_fn = self.run_streaming
            else:
                run_fn = self.run

            response = run_fn(user_input, verbose=verbose)
            if not stream:
                print(f"\nAgent: {response}\n")
