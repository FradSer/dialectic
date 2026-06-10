"""Step definitions for features/engine.feature — mocked LLM, no network."""

import asyncio
from unittest.mock import patch

from helpers import make_constant_call_agent
from pytest_bdd import given, parsers, scenarios, then, when

from dialectica.agent import create_coordinator

scenarios("features/engine.feature")


@given(
    parsers.parse(
        "the default pipeline with max depth {depth:d} and beam width {width:d}"
    ),
    target_fixture="coordinator",
)
def coordinator(depth: int, width: int):
    return create_coordinator(
        problem="How do we test the default composition?",
        max_depth=depth,
        beam_width=width,
        max_gan_rounds=1,
        score_threshold=7.0,
    )


@given(parsers.parse("every thought is scored {score:g}"), target_fixture="fake_llm")
def fake_llm(score: float):
    return make_constant_call_agent(score)


@when("the engine runs", target_fixture="result")
def run_engine(coordinator, fake_llm):
    with patch("dialectica.agent_runtime.run_agent", fake_llm):
        return asyncio.run(coordinator.run())


@then(parsers.parse('the final answer is "{answer}"'))
def final_answer_is(result, answer: str):
    assert result["final_answer"] == answer


@then(parsers.parse("the tree contains {count:d} thoughts"))
def tree_size_is(result, count: int):
    assert result["stats"]["total_thoughts"] == count


@then("the best path starts at the root")
def best_path_starts_at_root(result):
    assert result["best_path"][0] == "root"
    assert len(result["best_path"]) >= 2


@then("the beam is empty")
def beam_is_empty(coordinator):
    assert coordinator.active_beam == []
