from typing import Literal, Optional

from pydantic import BaseModel, Field


JokeQuality = Literal["Pass", "Fail"]

JOKE_RETRY_LIMIT = 3


class JokeFlowState(BaseModel):
    """State that is private to the joke refinement flow."""

    draft: Optional[str] = None
    final: Optional[str] = None
    style_hint: Optional[str] = None
    quality: Optional[JokeQuality] = None
    attempts: int = 0


class State(BaseModel):
    topic: str
    final_joke: Optional[str] = None
    poem: Optional[str] = None
    story: Optional[str] = None
    joke_flow: JokeFlowState = Field(default_factory=JokeFlowState)


class JokeQualityDecision(BaseModel):
    result: JokeQuality
