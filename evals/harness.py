"""Eval orchestration: run engine and baseline per problem, judge blind, report.

LLM calls are counted through the single runtime seam
(``dialectica.agent_runtime.run_agent``), so cost is measured the same way the
mocked tests intercept it — judge calls are deliberately not counted against
either contender.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from pydantic import BaseModel, Field

from dialectica import agent_runtime
from dialectica.coordinator import Coordinator

from .baseline import SingleCallBaseline
from .judge import BlindJudge
from .problems import EvalProblem

logger = logging.getLogger(__name__)


class CallCounter:
    """Counts LLM calls made while a ``count_agent_calls`` block is active."""

    def __init__(self):
        self.count = 0


@contextmanager
def count_agent_calls() -> Iterator[CallCounter]:
    """Count every ``run_agent`` call made inside the block.

    Wraps whatever is currently installed at the seam (the real runner or a
    test fake) and restores it on exit.
    """
    counter = CallCounter()
    original = agent_runtime.run_agent

    async def counting_run_agent(agent, instruction: str) -> str:
        counter.count += 1
        return await original(agent, instruction)

    agent_runtime.run_agent = counting_run_agent
    try:
        yield counter
    finally:
        agent_runtime.run_agent = original


@contextmanager
def retry_agent_calls(max_attempts: int = 3, base_delay: float = 2.0) -> Iterator[None]:
    """Retry transient ``run_agent`` failures with exponential backoff.

    An eval run is hundreds of sequential LLM calls; without this, a single
    transient network error or rate limit throws away the whole run. The last
    failure is re-raised so persistent errors still surface.
    """
    original = agent_runtime.run_agent

    async def retrying_run_agent(agent, instruction: str) -> str:
        for attempt in range(1, max_attempts + 1):
            try:
                return await original(agent, instruction)
            except Exception as e:
                if attempt == max_attempts:
                    raise
                delay = base_delay * 2 ** (attempt - 1)
                logger.warning(
                    "Agent call failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt,
                    max_attempts,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)

    agent_runtime.run_agent = retrying_run_agent
    try:
        yield
    finally:
        agent_runtime.run_agent = original


class ProblemResult(BaseModel):
    """Engine-vs-baseline outcome for one benchmark problem."""

    problem_id: str
    problem: str
    engine_answer: str
    baseline_answer: str
    engine_calls: int = Field(..., ge=0)
    baseline_calls: int = Field(..., ge=0)
    engine_seconds: float = Field(..., ge=0)
    baseline_seconds: float = Field(..., ge=0)
    winner: str = Field(..., description='"engine", "baseline", or "tie".')
    judge_reasoning: list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    """All per-problem results plus the aggregate tally."""

    results: list[ProblemResult]
    engine_wins: int
    baseline_wins: int
    ties: int

    @classmethod
    def from_results(cls, results: list[ProblemResult]) -> "EvalReport":
        tally = {"engine": 0, "baseline": 0, "tie": 0}
        for result in results:
            tally[result.winner] += 1
        return cls(
            results=results,
            engine_wins=tally["engine"],
            baseline_wins=tally["baseline"],
            ties=tally["tie"],
        )


async def run_eval(
    problems: list[EvalProblem],
    *,
    engine_factory: Callable[[str], Coordinator],
    baseline: SingleCallBaseline,
    judge: BlindJudge,
) -> EvalReport:
    """Run every problem through the engine and the baseline, then judge blind."""
    results: list[ProblemResult] = []
    with retry_agent_calls():
        for problem in problems:
            with count_agent_calls() as engine_counter:
                engine_start = time.perf_counter()
                engine_run = await engine_factory(problem.statement).run()
                engine_seconds = time.perf_counter() - engine_start
            engine_answer = engine_run["final_answer"]

            with count_agent_calls() as baseline_counter:
                baseline_start = time.perf_counter()
                baseline_answer = await baseline.answer(problem.statement)
                baseline_seconds = time.perf_counter() - baseline_start

            comparison = await judge.compare(
                problem.statement, engine_answer, baseline_answer
            )

            results.append(
                ProblemResult(
                    problem_id=problem.id,
                    problem=problem.statement,
                    engine_answer=engine_answer,
                    baseline_answer=baseline_answer,
                    engine_calls=engine_counter.count,
                    baseline_calls=baseline_counter.count,
                    engine_seconds=engine_seconds,
                    baseline_seconds=baseline_seconds,
                    winner=comparison.winner,
                    judge_reasoning=[v.reasoning for v in comparison.verdicts],
                )
            )

    return EvalReport.from_results(results)


def render_markdown(report: EvalReport) -> str:
    """Render the report as a human-readable Markdown summary."""
    lines = [
        "# Dialectica eval: engine vs single-call baseline",
        "",
        f"**Engine wins: {report.engine_wins} · "
        f"Baseline wins: {report.baseline_wins} · Ties: {report.ties}**",
        "",
        "| Problem | Winner | Engine calls | Baseline calls | Engine s | Baseline s |",
        "|---------|--------|--------------|----------------|----------|------------|",
    ]
    for r in report.results:
        lines.append(
            f"| {r.problem_id} | {r.winner} | {r.engine_calls} "
            f"| {r.baseline_calls} | {r.engine_seconds:.1f} | {r.baseline_seconds:.1f} |"
        )
    lines.append("")
    for r in report.results:
        lines.append(f"## {r.problem_id} — winner: {r.winner}")
        for i, reasoning in enumerate(r.judge_reasoning, 1):
            if reasoning:
                lines.append(f"- Judge pass {i}: {reasoning}")
        lines.append("")
    return "\n".join(lines)
