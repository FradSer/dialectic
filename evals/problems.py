"""Benchmark problems for the eval harness."""

from pydantic import BaseModel, Field


class EvalProblem(BaseModel):
    """One benchmark problem."""

    id: str = Field(..., description="Short slug identifying the problem.")
    statement: str = Field(..., min_length=1, description="The problem statement.")


DEFAULT_PROBLEMS = [
    EvalProblem(
        id="cloud-costs",
        statement=(
            "A 5-person startup spends $40k/month on cloud infrastructure for a "
            "B2B SaaS with 2,000 customers. Propose a concrete plan to cut the "
            "bill by half within one quarter without hurting reliability."
        ),
    ),
    EvalProblem(
        id="api-versioning",
        statement=(
            "Design a versioning and deprecation strategy for a public REST API "
            "consumed by 500+ third-party integrations, minimizing breakage "
            "while allowing the schema to evolve."
        ),
    ),
    EvalProblem(
        id="flaky-tests",
        statement=(
            "A 400-test CI suite fails 1 run in 5 due to flaky tests, and "
            "engineers have started ignoring red builds. Diagnose the likely "
            "causes and lay out a remediation plan with measurable milestones."
        ),
    ),
    EvalProblem(
        id="meeting-overload",
        statement=(
            "An engineering org of 60 people averages 18 hours of meetings per "
            "person per week. Design an intervention that recovers at least 6 "
            "hours per person without losing alignment."
        ),
    ),
    EvalProblem(
        id="urban-transport",
        statement=(
            "Design a sustainable urban transport plan for a mid-size city of "
            "800k people with a limited budget, balancing emissions, equity, "
            "and political feasibility."
        ),
    ),
]
