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

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner

logger = logging.getLogger(__name__)


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

    Rate-limit errors (429/RESOURCE_EXHAUSTED) wait ``RATE_LIMIT_COOLDOWN``
    per attempt so the quota window can roll over; other failures use fast
    exponential backoff.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return await _call_agent_once(agent, instruction)
        except Exception as e:
            if attempt == max_attempts:
                raise
            if _is_rate_limited(e):
                delay = RATE_LIMIT_COOLDOWN * attempt
            else:
                delay = base_delay * 2 ** (attempt - 1)
            logger.warning(
                "Agent call failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt,
                max_attempts,
                delay,
                e,
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")
