"""Unit tests for the role-template agent factory (no network)."""

from dialectica.agent_factory import ROLE_TEMPLATES, create_agent
from dialectica.models import DiscriminatorVerdict


def test_known_roles_have_templates():
    assert set(ROLE_TEMPLATES) == {"Generator", "Discriminator", "Synthesizer"}


def test_create_agent_uses_role_template_and_name():
    agent = create_agent(role="Discriminator", role_name="Critic")
    assert agent.name == "Critic"
    assert "Critic" in agent.instruction


def test_unknown_role_falls_back_to_generator():
    agent = create_agent(role="Nonexistent", role_name="X")
    assert "generating high-quality thoughts" in agent.instruction


def test_output_schema_forces_structured_output_and_drops_tools():
    agent = create_agent(
        role="Discriminator",
        role_name="Discriminator",
        tools=[lambda: None],
        output_schema=DiscriminatorVerdict,
    )
    assert agent.output_schema is DiscriminatorVerdict
    assert agent.tools == []


def test_additional_context_is_injected():
    agent = create_agent(
        role="Generator", role_name="Generator", additional_context="EXTRA RULES"
    )
    assert "EXTRA RULES" in agent.instruction
