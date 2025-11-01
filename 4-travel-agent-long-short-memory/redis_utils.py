import os
import logging
logger = logging.getLogger(__name__)

from redis import Redis
from redisvl.index import SearchIndex
from redisvl.schema.schema import IndexSchema
from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.checkpoint.redis import RedisSaver


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

redis_client = Redis.from_url(REDIS_URL)
logger.info(f"Ping {redis_client.ping()}")

@cache
def redis_saver():
    redis_saver = RedisSaver(redis_client=redis_client)
    redis_saver.setup()

@cache
def init_redis_index():
    memory_schema = IndexSchema.from_dict({
            "index": {
                "name": "agent_memories",  # Index name for identification
                "prefix": "memory",       # Redis key prefix (memory:1, memory:2, etc.)
                "key_separator": ":",
                "storage_type": "json",
            },
            "fields": [
                {"name": "content", "type": "text"},
                {"name": "memory_type", "type": "tag"},
                {"name": "metadata", "type": "text"},
                {"name": "created_at", "type": "text"},
                {"name": "user_id", "type": "tag"},
                {"name": "memory_id", "type": "tag"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": 1536,  # OpenAI embedding dimension
                        "distance_metric": "cosine",
                        "datatype": "float32",
                    },
                },
            ],
        }
    )
    try:
        long_term_memory_index = SearchIndex(
            schema=memory_schema,
            redis_client=redis_client,
            validate_on_load=True
        )
        long_term_memory_index.create(overwrite=True)
        logger.info("Long-term memory index ready")
    except Exception as e:
        logger.info(f"Error creating index: {e}")
        return None
    return long_term_memory_index

