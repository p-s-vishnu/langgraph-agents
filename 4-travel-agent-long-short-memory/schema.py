import ulid

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """
    EPISODIC: Personal experiences and user-specific preferences
              (e.g., "User prefers Delta airlines", "User visited Paris last year")

    SEMANTIC: General domain knowledge and facts
              (e.g., "Singapore requires passport", "Tokyo has excellent public transit")
    """

    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class Memory(BaseModel):
    content: str
    memory_type: MemoryType
    metadata: str


class Memories(BaseModel):
    """
    NOTE: OpenAI's structured output requires us to wrap the list in an object.
    """
    memories: List[Memory]


class StoredMemory(Memory):
    """A stored long-term memory"""
    id: str  # The redis key
    memory_id: ulid.ULID = Field(default_factory=lambda: ulid.ULID())
    created_at: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    memory_type: Optional[MemoryType] = None
