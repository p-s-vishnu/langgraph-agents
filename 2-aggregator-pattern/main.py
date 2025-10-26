import anyio
import fastenv
import asyncio
import os
from typing import Any, Dict


def configure_tracing_defaults() -> None:
    """Ensure LangSmith traces land in the expected project."""
    os.environ.setdefault("LANGCHAIN_PROJECT", "agents")

async def load_env_async():
    dotenv = await fastenv.load_dotenv("2-aggregator-pattern/.env")
    os.environ.update(dict(dotenv))

async def invoke_graph(topic: str) -> Dict[str, Any]:
    """Run the compiled graph for a single topic."""
    # Import lazily so configuration above happens first in CLI runs.
    from agent import aggregator_app

    # TODO: async call to draw graph
    aggregator_app.get_graph().draw_mermaid_png(
        output_file_path="2-aggregator-pattern/compiled_graph.png"
    )
    return await aggregator_app.ainvoke({"topic": topic})


if __name__ == "__main__":
    anyio.run(load_env_async)
    configure_tracing_defaults()
    topic = input("Enter a joke topic: ").strip()
    result = asyncio.run(invoke_graph(topic))
