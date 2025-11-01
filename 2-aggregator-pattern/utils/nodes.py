import asyncio
import random
import time
from collections import deque
from typing import Any, Dict

import aiofiles
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from utils.state import (
    JOKE_RETRY_LIMIT,
    JokeFlowState,
    JokeQualityDecision,
    State,
)


# Generic functions
class AsyncRateLimiter:
    """Simple async-friendly rate limiter supporting burst control."""

    def __init__(self, max_calls: int, period_seconds: float):
        self._max_calls = max_calls
        self._period = period_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self._period:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max_calls:
                    self._timestamps.append(now)
                    return
                wait_time = self._period - (now - self._timestamps[0])
            await asyncio.sleep(max(wait_time, 0.0))


_model_cache: Dict[str, Any] = {}
_model_lock = asyncio.Lock()
_rate_limiters = {
    "openai": AsyncRateLimiter(max_calls=3, period_seconds=1),
    "google": AsyncRateLimiter(max_calls=3, period_seconds=1),
}


async def _get_model(provider: str):
    """Create or return the cached chat model for a provider."""
    async with _model_lock:
        cached = _model_cache.get(provider)
        if cached is not None:
            return cached
        if provider == "openai":
            model = ChatOpenAI(model="gpt-4.1", temperature=0.9)
        elif provider == "google":
            model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                temperature=0.8,
            )
        else:
            raise ValueError(f"Unsupported model provider: {provider}")
        _model_cache[provider] = model
        return model


async def _extract_text(response: Any) -> str:
    """Normalize the response payload from LangChain chat models."""
    if response is None:
        return ""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return " ".join(part["text"] for part in content if "text" in part).strip()
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _next_joke_flow(state: State, **updates: Any) -> JokeFlowState:
    """Return a new joke flow state merging the provided updates."""
    current = state.joke_flow.model_copy(deep=True)
    return current.model_copy(update=updates)

# Flow 1: Joke workflow nodes
async def generate_joke(state: State) -> Dict[str, Any]:
    """Create the first version of a joke."""
    llm = await _get_model("openai")
    style = random.choice(["deadpan", "absurdist", "wordplay-heavy", "self-aware stand-up routine", "wholesome storytelling"])
    prompt = (
        "Invent a brand-new joke that has not been told before. "
        f"Topic: {state.topic}. Work in a {style} style and avoid any known "
        "classics such as 'cat on the computer' or 'keeping an eye on the mouse'. "
        "Keep it to 2-3 sentences."
    )
    async with _rate_limiters["openai"]:
        response = await llm.ainvoke(prompt)
    joke_text = await _extract_text(response)
    return {
        "joke_flow": _next_joke_flow(
            state,
            draft=joke_text,
            style_hint=style,
            attempts=1,
            quality=None,
            final=None,
        )
    }


async def check_joke_quality(state: State) -> Dict[str, Any]:
    """Decide whether the initial joke passes the funniness threshold."""
    if state.joke_flow.draft is None:
        raise ValueError("No joke is available to evaluate.")
    llm = (await _get_model("google")).with_structured_output(JokeQualityDecision)
    prompt = (
        "Evaluate the joke below. Respond with result='Pass' if it's funny enough, "
        "otherwise use result='Fail'. Joke: "
        f"{state.joke_flow.draft}"
    )
    async with _rate_limiters["google"]:
        assessment = await llm.ainvoke(prompt)
    decision = getattr(assessment, "result", None)
    if decision is None and isinstance(assessment, dict):
        decision = assessment.get("result")
    if decision not in {"Pass", "Fail"}:
        raise ValueError(f"Unexpected evaluation response: {assessment!r}")
    return {
        "joke_flow": _next_joke_flow(
            state,
            quality=decision,
        )
    }


async def route_joke_quality(state: State) -> str:
    if state.joke_flow.quality is None:
        raise ValueError("Missing joke quality decision on state.")
    if state.joke_flow.quality == "Pass":
        return "passed"
    if state.joke_flow.attempts < JOKE_RETRY_LIMIT:
        return "retry"
    return "give_up"


async def improve_joke(state: State) -> Dict[str, Any]:
    if state.joke_flow.draft is None:
        raise ValueError("No joke is available to improve.")
    if state.joke_flow.attempts >= JOKE_RETRY_LIMIT:
        return {"joke_flow": state.joke_flow}
    llm = await _get_model("google")
    prompt = (
        "Punch up the following joke with clever wordplay while keeping it concise. "
        "Preserve the core premise but change the structure so it feels fresh. "
        f"Target style: {state.joke_flow.style_hint or 'surprising and playful'}. "
        f"Joke: {state.joke_flow.draft}"
    )
    async with _rate_limiters["google"]:
        response = await llm.ainvoke(prompt)
    improved_text = await _extract_text(response)
    return {
        "joke_flow": _next_joke_flow(
            state,
            draft=improved_text,
            attempts=state.joke_flow.attempts + 1,
            quality=None,
        )
    }


async def finalize_joke(state: State) -> Dict[str, Any]:
    setup = state.joke_flow.draft
    if setup is None:
        raise ValueError("No joke is available to finalize.")
    llm = await _get_model("openai")
    prompt = (
        "Provide the final version of this joke. Retain the core idea but add a "
        f"surprising twist. Style guide: {state.joke_flow.style_hint or 'inventive'}. "
        f"Draft: {setup}"
    )
    async with _rate_limiters["openai"]:
        response = await llm.ainvoke(prompt)
    final_text = await _extract_text(response)
    updated_flow = _next_joke_flow(
        state,
        final=final_text,
    )
    return {"final_joke": final_text, "joke_flow": updated_flow}

# Flow 2: Story workflow nodes
async def write_story(state: State) -> Dict[str, str]:
    if state.topic is None:
        raise ValueError("No topic is available to write a story about.")
    llm = await _get_model("openai")
    prompt = (
        "Write a short story based on the following topic. Make it engaging "
        "and imaginative. Topic: "
        f"{state.topic}"
    )
    async with _rate_limiters["openai"]:
        response = await llm.ainvoke(prompt)
    return {"story": await _extract_text(response)}

# Flow 3: Poem workflow nodes
async def compose_poem(state: State) -> Dict[str, str]:
    if state.topic is None:
        raise ValueError("No topic is available to compose a poem about.")
    llm = await _get_model("google")
    prompt = (
        "Compose a short poem inspired by the following topic. Use vivid imagery "
        "and rhythmic language. Topic: "
        f"{state.topic}"
    )
    async with _rate_limiters["google"]:
        response = await llm.ainvoke(prompt)
    return {"poem": await _extract_text(response)}

# Aggregator node
async def aggregate_answers(state: State) -> Dict[str, Any]:
    """Combine outputs from multiple flows into a single payload and save to a markdown file in async manner."""
    combined = {
        "joke": state.final_joke or state.joke_flow.final or state.joke_flow.draft,
        "story": state.story,
        "poem": state.poem,
    }

    combined = {k: v for k, v in combined.items() if v is not None}
    # Save to markdown file
    async with aiofiles.open("2-aggregator-pattern/output.md", "w") as f:
        await f.write(f"# Aggregated Creations for `{state.topic}`\n\n")
        for key, value in combined.items():
            await f.write(f"## {key.title()}\n\n{value}\n\n")
    return combined
