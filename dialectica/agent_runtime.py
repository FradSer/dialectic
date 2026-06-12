"""Single entry point for invoking an LlmAgent.

Centralizing the ADK Runner call gives every pluggable component (generator,
evaluator, synthesizer) one shared seam — which is also the one place tests
patch to run the engine without the network.

``run_agent`` retries transient failures with exponential backoff: an engine
run is hundreds of sequential LLM calls, and without retry a single network
error or rate limit throws the whole run away. Persistent failures re-raise.
"""

import asyncio
import logging
import os
import random

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner

logger = logging.getLogger(__name__)

# Optional global cap on concurrent LLM calls, for tightly-quota'd backends
# (e.g. gemma-4-31b allows only 16k input tokens/minute — unbounded gather
# self-collides on the quota). 0 or unset = unlimited.
_concurrency_limiter: asyncio.Semaphore | None = None
_limiter_configured = False


def _reset_concurrency_limiter() -> None:
    """Re-read DIALECTICA_MAX_CONCURRENCY on next call (used by tests)."""
    global _concurrency_limiter, _limiter_configured
    _concurrency_limiter = None
    _limiter_configured = False


def _get_concurrency_limiter() -> asyncio.Semaphore | None:
    global _concurrency_limiter, _limiter_configured
    if not _limiter_configured:
        _limiter_configured = True
        cap = int(os.environ.get("DIALECTICA_MAX_CONCURRENCY", "0") or "0")
        if cap > 0:
            _concurrency_limiter = asyncio.Semaphore(cap)
    return _concurrency_limiter


async def _call_agent_once(agent: LlmAgent, instruction: str) -> str:
    """One raw ADK invocation, returning the concatenated text output."""
    runner = InMemoryRunner(agent=agent, app_name="dialectica")
    events = await runner.run_debug(instruction, quiet=True)

    response_text = ""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text and not part.thought:
                    response_text += part.text

    return response_text.strip()


# Rate-limit quotas (e.g. tokens-per-minute) need the window to roll over;
# exponential backoff in seconds just burns the remaining attempts.
RATE_LIMIT_COOLDOWN = 45.0
MAX_RATE_LIMIT_RETRIES = 8

_RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "rate limit", "RateLimit")


def _is_rate_limited(error: Exception) -> bool:
    text = str(error)
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


async def run_agent(
    agent: LlmAgent,
    instruction: str,
    *,
    max_attempts: int = 3,
    base_delay: float = 2.0,
) -> str:
    """Run ``agent`` on ``instruction``, retrying transient failures.

    Rate-limit errors (429/RESOURCE_EXHAUSTED) have their own retry budget
    (``MAX_RATE_LIMIT_RETRIES``) and wait ``RATE_LIMIT_COOLDOWN`` (scaled,
    jittered to desynchronize concurrent callers) so the quota window can
    roll over; other failures use ``max_attempts`` with fast exponential
    backoff. ``DIALECTICA_MAX_CONCURRENCY`` caps overlapping calls globally.
    """
    limiter = _get_concurrency_limiter()
    failures = 0
    rate_limit_hits = 0
    while True:
        try:
            if limiter is not None:
                async with limiter:
                    return await _call_agent_once(agent, instruction)
            return await _call_agent_once(agent, instruction)
        except Exception as e:
            if _is_rate_limited(e):
                rate_limit_hits += 1
                if rate_limit_hits > MAX_RATE_LIMIT_RETRIES:
                    raise
                delay = RATE_LIMIT_COOLDOWN * min(rate_limit_hits, 3)
                delay += random.uniform(0, 10)
                logger.warning(
                    "Rate limited (hit %d/%d), cooling down %.1fs",
                    rate_limit_hits,
                    MAX_RATE_LIMIT_RETRIES,
                    delay,
                )
            else:
                failures += 1
                if failures >= max_attempts:
                    raise
                delay = base_delay * 2 ** (failures - 1)
                logger.warning(
                    "Agent call failed (attempt %d/%d), retrying in %.1fs: %s",
                    failures,
                    max_attempts,
                    delay,
                    e,
                )
            await asyncio.sleep(delay)
