"""Functions to access memories"""

import logging
from datetime import datetime
from typing import List, Optional, Union

from redisvl.utils.vectorize.text.openai import OpenAITextVectorizer
from redisvl.query import VectorRangeQuery
from redisvl.query.filter import Tag
import ulid

from schema import MemoryType, StoredMemory
from redis_utils import init_redis_index


# If we have any memories that aren't associated with a user, we'll use this ID.
SYSTEM_USER_ID = "system"
logger = logging.getLogger(__name__)
long_term_memory_index = init_redis_index()
openai_embed = OpenAITextVectorizer(model="text-embedding-ada-002")


def similar_memory_exists(
    content: str,
    memory_type: MemoryType,
    user_id: str = SYSTEM_USER_ID,
    thread_id: Optional[str] = None,
    distance_threshold: float = 0.1,
) -> bool:
    """Check if a similar long-term memory already exists in Redis."""
    content_embedding = openai_embed.embed(content)
    filters = (Tag("user_id") == user_id) & (Tag("memory_type") == memory_type)

    if thread_id:
        filters = filters & (Tag("thread_id") == thread_id)

    # Search for similar memories
    vector_query = VectorRangeQuery(
        vector=content_embedding,
        num_results=1,
        vector_field_name="embedding",
        filter_expression=filters,
        distance_threshold=distance_threshold,
        return_fields=["id"],
    )
    results = long_term_memory_index.query(vector_query)
    logger.debug(f"Similar memory search results: {results}")

    if results:
        logger.debug(
            f"{len(results)} similar {'memory' if results.count == 1 else 'memories'} found. First: "
            f"{results[0]['id']}. Skipping storage."
        )
        return True

    return False

def store_memory(
    content: str,
    memory_type: MemoryType,
    user_id: str = SYSTEM_USER_ID,
    thread_id: Optional[str] = None,
    metadata: Optional[str] = None,
):
    """Store a long-term memory in Redis with deduplication.

        This function:
        1. Checks for similar existing memories to avoid duplicates
        2. Generates vector embeddings for semantic search
        3. Stores the memory with metadata for retrieval
        """
    if metadata is None:
        metadata = "{}"

    logger.info(f"Preparing to store memory: {content}")

    if similar_memory_exists(content, memory_type, user_id, thread_id):
        logger.info("Similar memory found, skipping storage")
        return

    embedding = openai_embed.embed(content)
    memory_data = {
        "user_id": user_id or SYSTEM_USER_ID,
        "content": content,
        "memory_type": memory_type.value,
        "metadata": metadata,
        "created_at": datetime.now().isoformat(),
        "embedding": embedding,
        "memory_id": str(ulid.ULID()),
        "thread_id": thread_id,
    }

    try:
        long_term_memory_index.load([memory_data])
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        return
    logger.info(f"Stored {memory_type} memory: {content}")

def retrieve_memories(
    query: str,
    memory_type: Union[Optional[MemoryType], List[MemoryType]] = None,
    user_id: str = SYSTEM_USER_ID,
    thread_id: Optional[str] = None,
    distance_threshold: float = 0.1,
    limit: int = 5,
) -> List[StoredMemory]:
    logger.debug(f"Retrieving memories for query: {query}")
    vector_query = VectorRangeQuery(
        vector=openai_embed.embed(query),
        return_fields=[
            "content",
            "memory_type", 
            "metadata",
            "created_at",
            "memory_id",
            "thread_id",
            "user_id",
        ],
        num_results=limit,
        vector_field_name="embedding",
        dialect=2,
        distance_threshold=distance_threshold,
    )


    base_filters = [f"@user_id:{{{user_id or SYSTEM_USER_ID}}}"]
    if memory_type:
        if isinstance(memory_type, list):
            base_filters.append(f"@memory_type:{{{'|'.join(memory_type)}}}")
        else:
            base_filters.append(f"@memory_type:{{{memory_type.value}}}")
    if thread_id:
        base_filters.append(f"@thread_id:{{{thread_id}}}")

    vector_query.set_filter(" ".join(base_filters))
    similar_results = long_term_memory_index.query(vector_query)

    memories = []
    for doc in similar_results:
        try:
            memory = StoredMemory(
                id=doc["id"],
                memory_id=doc["memory_id"],
                user_id=doc["user_id"],
                thread_id=doc.get("thread_id", None),
                memory_type=MemoryType(doc["memory_type"]),
                content=doc["content"],
                created_at=doc["created_at"],
                metadata=doc["metadata"],
            )
            memories.append(memory)
        except Exception as e:
            logger.error(f"Error parsing memory: {e}")
    return memories
