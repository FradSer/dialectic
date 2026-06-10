"""Step definitions for features/adversarial_evaluation.feature — mocked LLM."""

import asyncio
from unittest.mock import patch

from helpers import make_call_agent
from pytest_bdd import given, parsers, scenarios, then, when

from dialectica.agent_factory import create_agent
from dialectica.gan_evaluator import AdversarialEvaluator
from dialectica.models import DiscriminatorVerdict

scenarios("features/adversarial_evaluation.feature")


@given(
    parsers.parse(
        "an adversarial evaluator with max rounds {rounds:d} and score threshold {threshold:g}"
    ),
    target_fixture="evaluator",
)
def evaluator(rounds: int, threshold: float):
    generator = create_agent(role="Generator", role_name="Generator")
    discriminator = create_agent(
        role="Discriminator",
        role_name="Discriminator",
        output_schema=DiscriminatorVerdict,
    )
    return AdversarialEvaluator(
        generator=generator,
        discriminator=discriminator,
        max_rounds=rounds,
        score_threshold=threshold,
    )


@given(
    parsers.parse('the discriminator returns scores "{scores}"'),
    target_fixture="llm_spec",
)
def llm_spec_from_scores(scores: str):
    verdicts = [{"score": float(s)} for s in scores.split(",")]
    return {"verdicts": verdicts, "refined": "A refined, stronger thought."}


@given(
    parsers.parse("the discriminator returns score {score:g} with termination"),
    target_fixture="llm_spec",
)
def llm_spec_with_termination(score: float):
    return {
        "verdicts": [{"score": score, "should_terminate": True}],
        "refined": "A refined, stronger thought.",
    }


@given(parsers.parse('the generator refines thoughts to "{refined}"'))
def generator_refines_to(llm_spec, refined: str):
    llm_spec["refined"] = refined


@when(parsers.parse('the evaluator judges "{thought}"'), target_fixture="result")
def judge(evaluator, llm_spec, thought: str):
    fake = make_call_agent(llm_spec["verdicts"], refined=llm_spec["refined"])
    with patch("dialectica.agent_runtime.run_agent", fake):
        return asyncio.run(evaluator.evaluate(thought, {"problem": "p"}))


@then(parsers.parse("the result score is {score:g}"))
def result_score_is(result, score: float):
    assert result.score == score


@then(parsers.parse("the loop ran {rounds:d} round"))
@then(parsers.parse("the loop ran {rounds:d} rounds"))
def loop_ran_rounds(result, rounds: int):
    assert result.adversarial_rounds == rounds
    assert len(result.history) == rounds


@then(parsers.parse('the refined thought is "{refined}"'))
def refined_thought_is(result, refined: str):
    assert result.refined_thought == refined


@then("the evaluation requests termination")
def evaluation_requests_termination(result):
    assert result.should_terminate is True
