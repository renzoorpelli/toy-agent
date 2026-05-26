from dotenv import load_dotenv

from .agent import Agent
from .context import Context
from .llm import LLM
from .store import VectorStore
from .tools import create_default_tools


def main() -> None:
    load_dotenv()
    llm = LLM()
    context = Context(system_prompt="You are a helpful AI assistant. Use tools when they can help answer the user's question.")

    store = VectorStore()
    tools = create_default_tools(store=store, llm=llm)

    agent = Agent(llm=llm, context=context, tools=tools)
    agent.chat(stream=True)


if __name__ == "__main__":
    main()