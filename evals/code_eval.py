"""Ground-truth code eval: engine vs baseline on HumanEval-style problems.

Verification executes the candidate implementation plus the problem's assert
tests in a subprocess with a timeout — pass/fail is objective, no LLM judge.
This runs model-generated code on the local machine, which is the standard
trade-off for HumanEval-style harnesses; problems are self-contained pure
functions and the subprocess gets a hard timeout.
"""

import asyncio
import re
import subprocess
import sys
import time
from collections.abc import Callable

from pydantic import BaseModel, Field

from dialectica.coordinator import Coordinator

from .baseline import SingleCallBaseline
from .code_problems import CodeProblem
from .harness import count_agent_calls

# Code-focused discriminator rubric (the criteria steer answer content).
CODE_CRITERIA = """\
1. **Correctness**: Does the approach satisfy the specification and the
   docstring examples exactly, including edge cases (empty input, negatives,
   boundaries)?
2. **Completeness**: Is it a full, runnable implementation — no pseudocode,
   no missing helpers?
3. **Robustness**: Does it terminate on all inputs and avoid off-by-one and
   type errors?
4. **Simplicity**: Prefer a direct, readable implementation over a clever
   but fragile one."""

STATEMENT_TEMPLATE = """Implement the following Python function correctly.

```python
{prompt}```

Requirements:
- Satisfy the docstring specification and examples exactly, including edge cases.
- The implementation must be complete and runnable as-is.

Output: a single ```python code block containing the full function definition
(plus any helpers you need). No commentary outside the code block."""

_PYTHON_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_ANY_BLOCK_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def build_statement(problem: CodeProblem) -> str:
    return STATEMENT_TEMPLATE.format(prompt=problem.prompt)


def extract_python_code(answer: str) -> str:
    """Extract the implementation from an answer.

    Prefers the last ```python block (final answers often restate the full
    implementation last), falls back to any fenced block, then to the raw
    text.
    """
    blocks = _PYTHON_BLOCK_RE.findall(answer) or _ANY_BLOCK_RE.findall(answer)
    if blocks:
        return blocks[-1].strip()
    return answer.strip()


class VerifyResult(BaseModel):
    """Outcome of executing one candidate against the problem's tests."""

    passed: bool
    output: str = Field("", description="Tail of stderr/stdout on failure.")


def verify_solution(
    problem: CodeProblem, code: str, timeout: float = 10.0
) -> VerifyResult:
    """Run ``code`` plus the problem's asserts in a subprocess."""
    program = f"{code}\n\n{problem.tests}"
    try:
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return VerifyResult(passed=False, output=f"timeout after {timeout}s")
    if completed.returncode == 0:
        return VerifyResult(passed=True)
    return VerifyResult(passed=False, output=(completed.stderr or "")[-500:])


class CodeProblemResult(BaseModel):
    """Engine-vs-baseline ground-truth outcome for one code problem."""

    problem_id: str
    engine_passed: bool
    baseline_passed: bool
    engine_error: str = ""
    baseline_error: str = ""
    engine_calls: int = Field(..., ge=0)
    baseline_calls: int = Field(..., ge=0)
    engine_seconds: float = Field(..., ge=0)
    baseline_seconds: float = Field(..., ge=0)


class CodeEvalReport(BaseModel):
    """All per-problem results plus pass-rate aggregates."""

    results: list[CodeProblemResult]
    engine_passed: int
    baseline_passed: int

    @classmethod
    def from_results(cls, results: list[CodeProblemResult]) -> "CodeEvalReport":
        return cls(
            results=results,
            engine_passed=sum(r.engine_passed for r in results),
            baseline_passed=sum(r.baseline_passed for r in results),
        )


async def run_code_eval(
    problems: list[CodeProblem],
    *,
    engine_factory: Callable[[str], Coordinator],
    baseline: SingleCallBaseline,
) -> CodeEvalReport:
    """Solve every problem with the engine and the baseline, verify both."""
    results: list[CodeProblemResult] = []
    for problem in problems:
        statement = build_statement(problem)

        with count_agent_calls() as engine_counter:
            engine_start = time.perf_counter()
            engine_run = await engine_factory(statement).run()
            engine_seconds = time.perf_counter() - engine_start
        engine_code = extract_python_code(engine_run["final_answer"])
        engine_verdict = await asyncio.to_thread(verify_solution, problem, engine_code)

        with count_agent_calls() as baseline_counter:
            baseline_start = time.perf_counter()
            baseline_answer = await baseline.answer(statement)
            baseline_seconds = time.perf_counter() - baseline_start
        baseline_code = extract_python_code(baseline_answer)
        baseline_verdict = await asyncio.to_thread(
            verify_solution, problem, baseline_code
        )

        results.append(
            CodeProblemResult(
                problem_id=problem.id,
                engine_passed=engine_verdict.passed,
                baseline_passed=baseline_verdict.passed,
                engine_error=engine_verdict.output,
                baseline_error=baseline_verdict.output,
                engine_calls=engine_counter.count,
                baseline_calls=baseline_counter.count,
                engine_seconds=engine_seconds,
                baseline_seconds=baseline_seconds,
            )
        )

    return CodeEvalReport.from_results(results)


def render_code_markdown(report: CodeEvalReport) -> str:
    """Render the code eval report as a Markdown summary."""
    total = len(report.results)
    lines = [
        "# Dialectica code eval: engine vs single-call baseline (ground truth)",
        "",
        f"**Engine: {report.engine_passed}/{total} passed · "
        f"Baseline: {report.baseline_passed}/{total} passed**",
        "",
        "| Problem | Engine | Baseline | Engine calls | Engine s | Baseline s |",
        "|---------|--------|----------|--------------|----------|------------|",
    ]
    for r in report.results:
        engine = "pass" if r.engine_passed else "FAIL"
        baseline = "pass" if r.baseline_passed else "FAIL"
        lines.append(
            f"| {r.problem_id} | {engine} | {baseline} "
            f"| {r.engine_calls} | {r.engine_seconds:.1f} | {r.baseline_seconds:.1f} |"
        )
    lines.append("")
    for r in report.results:
        for arm, passed, error in (
            ("engine", r.engine_passed, r.engine_error),
            ("baseline", r.baseline_passed, r.baseline_error),
        ):
            if not passed and error:
                lines.append(f"## {r.problem_id} — {arm} failure")
                lines.append(f"```\n{error}\n```")
                lines.append("")
    return "\n".join(lines)
