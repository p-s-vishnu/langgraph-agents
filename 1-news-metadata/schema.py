from typing import List, Optional

from pydantic import BaseModel, Field


class State(BaseModel):
    text: str
    classification: Optional[str] = None
    entities: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
