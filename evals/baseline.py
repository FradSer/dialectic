"""Single-call baseline: one direct answer from a strong model.

This is the control arm of the eval: the engine must beat what a single
well-prompted LLM call produces to justify its cost. Configure the model with
``BASELINE_MODEL_CONFIG`` (falls back to ``DEFAULT_MODEL_CONFIG``).
"""

from google.adk.agents import LlmAgent

from dialectica import agent_runtime
from dialectica.llm_config import get_model_config

BASELINE_SYSTEM_PROMPT = """You are an expert problem solver.

Give your single best answer to the problem: correct, complete, specific and
actionable. Structure the answer clearly with sections if appropriate."""

BASELINE_INSTRUCTION = """Solve the following problem:

**Problem:**
{problem}

**Output:**
Provide the solution directly, without additional commentary."""


def create_baseline_agent() -> LlmAgent:
    """Create the single-call baseline agent."""
    return LlmAgent(
        name="Baseline",
        instruction=BASELINE_SYSTEM_PROMPT,
        model=get_model_config("Baseline"),
    )


class SingleCallBaseline:
    """Answers a problem with exactly one LLM call."""

    def __init__(self, agent: LlmAgent):
        self.agent = agent

    async def answer(self, problem: str) -> str:
        instruction = BASELINE_INSTRUCTION.format(problem=problem)
        response = await agent_runtime.run_agent(self.agent, instruction)
        return response.strip()
