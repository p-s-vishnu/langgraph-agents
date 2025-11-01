from typing import Dict, Optional, List

from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig

from schema import MemoryType
from utils import store_memory, retrieve_memories, SYSTEM_USER_ID


@tool
def store_memory_tool(
    content: str,
    memory_type: MemoryType,
    metadata: Optional[Dict[str, str]] = None,
    config: Optional[RunnableConfig] = None,
) -> str:
    """
    Store a long-term memory in the system.

    Use this tool to save important information about user preferences,
    experiences, or general knowledge that might be useful in future
    interactions.
    """
    config = config or RunnableConfig()
    user_id = config.get("user_id", SYSTEM_USER_ID)
    thread_id = config.get("thread_id")

    try:
        store_memory(
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            thread_id=thread_id,
            metadata=str(metadata) if metadata else None,
        )
        return f"Successfully stored {memory_type} memory: {content}"
    except Exception as e:
        return f"Error storing memory: {str(e)}"
    
@tool
def retrieve_memories_tool(
    query: str,
    memory_type: List[MemoryType],
    limit: int = 5,
    config: Optional[RunnableConfig] = None,
) -> str:
    """
    Retrieve long-term memories relevant to the query.

    Use this tool to access previously stored information about user
    preferences, experiences, or general knowledge.
    """
    config = config or RunnableConfig()
    user_id = config.get("user_id", SYSTEM_USER_ID)

    try:
        # Get long-term memories
        stored_memories = retrieve_memories(
            query=query,
            memory_type=memory_type,
            user_id=user_id,
            limit=limit,
            distance_threshold=0.3,
        )

        # Format the response
        response = []

        if stored_memories:
            response.append("Long-term memories:")
            for memory in stored_memories:
                response.append(f"- [{memory.memory_type}] {memory.content}")

        return "\n".join(response) if response else "No relevant memories found."

    except Exception as e:
        return f"Error retrieving memories: {str(e)}"