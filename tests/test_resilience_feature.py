"""Step definitions for features/resilience.feature.

These exercise the retry loop inside ``agent_runtime.run_agent`` itself, so
they patch the one private transport seam below it (``_call_agent_once``)
rather than ``run_agent`` — patching ``run_agent`` would bypass the behavior
under test.
"""

import asyncio
from unittest.mock import patch

from pytest_bdd import given, parsers, scenarios, then, when

from dialectica import agent_runtime

scenarios("features/resilience.feature")


class FlakyTransport:
    """Async ``_call_agent_once`` stand-in failing ``failures`` times first."""

    def __init__(self, failures: int):
        self.failures = failures
        self.attempts = 0

    async def __call__(self, agent, instruction: str) -> str:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise ConnectionError("transient network failure")
        return "ok"


@given(
    parsers.parse("an LLM transport that fails {n:d} times before succeeding"),
    target_fixture="transport",
)
def flaky_transport(n: int):
    return FlakyTransport(failures=n)


@given("an LLM transport that always fails", target_fixture="transport")
def always_failing_transport():
    return FlakyTransport(failures=10**6)


class RateLimitedTransport:
    """Fails ``limited`` times with a 429-style error, then succeeds."""

    def __init__(self, limited: int = 1):
        self.limited = limited
        self.attempts = 0

    async def __call__(self, agent, instruction: str) -> str:
        self.attempts += 1
        if self.attempts <= self.limited:
            raise RuntimeError(
                "429 RESOURCE_EXHAUSTED. Quota exceeded; please retry in 42s."
            )
        return "ok"


@given(
    "an LLM transport that is rate limited once before succeeding",
    target_fixture="transport",
)
def rate_limited_transport():
    return RateLimitedTransport()


@given(
    parsers.parse(
        "an LLM transport that is rate limited {n:d} times before succeeding"
    ),
    target_fixture="transport",
)
def rate_limited_n_transport(n: int):
    return RateLimitedTransport(limited=n)


@when("an agent call runs through the runtime", target_fixture="outcome")
def run_call(transport):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float):
        sleeps.append(seconds)

    async def go():
        return await agent_runtime.run_agent(None, "x", max_attempts=3, base_delay=0)

    with (
        patch("dialectica.agent_runtime._call_agent_once", transport),
        patch("dialectica.agent_runtime.asyncio.sleep", fake_sleep),
    ):
        try:
            return {"result": asyncio.run(go()), "error": None, "sleeps": sleeps}
        except ConnectionError as e:
            return {"result": None, "error": e, "sleeps": sleeps}


@then(parsers.parse("the call succeeds after {n:d} attempts"))
def call_succeeds(outcome, transport, n: int):
    assert outcome["error"] is None
    assert outcome["result"] == "ok"
    assert transport.attempts == n


@then(parsers.parse("the call fails after exhausting {n:d} attempts"))
def call_fails(outcome, transport, n: int):
    assert outcome["result"] is None
    assert isinstance(outcome["error"], ConnectionError)
    assert transport.attempts == n


@then(parsers.parse("the retry waited at least {seconds:d} seconds"))
def retry_waited(outcome, seconds: int):
    assert outcome["sleeps"]
    assert max(outcome["sleeps"]) >= seconds


class OverlapProbe:
    """Transport recording the maximum number of overlapping calls."""

    def __init__(self):
        self.in_flight = 0
        self.max_in_flight = 0

    async def __call__(self, agent, instruction: str) -> str:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            return "ok"
        finally:
            self.in_flight -= 1


@given(
    parsers.parse("the runtime concurrency cap is {n:d}"),
    target_fixture="capped_runtime",
)
def capped_runtime(n: int, monkeypatch):
    monkeypatch.setenv("DIALECTICA_MAX_CONCURRENCY", str(n))
    agent_runtime._reset_concurrency_limiter()
    yield n
    monkeypatch.delenv("DIALECTICA_MAX_CONCURRENCY", raising=False)
    agent_runtime._reset_concurrency_limiter()


@when(
    parsers.parse("{n:d} agent calls run concurrently through the runtime"),
    target_fixture="overlap_probe",
)
def run_concurrent_calls(capped_runtime, n: int):
    probe = OverlapProbe()

    async def go():
        await asyncio.gather(*(agent_runtime.run_agent(None, "x") for _ in range(n)))

    with patch("dialectica.agent_runtime._call_agent_once", probe):
        asyncio.run(go())
    return probe


@then(parsers.parse("no more than {n:d} call was in flight at once"))
def max_overlap_is(overlap_probe, n: int):
    assert overlap_probe.max_in_flight <= n
