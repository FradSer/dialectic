"""
Dialectic — a pluggable adversarial reasoning engine.

The Engine runs a beam-style tree search and delegates each stage to a
swappable component (Generator / Evaluator / Selector / Synthesizer):
thesis -> antithesis -> synthesis. The defaults give a Tree-of-Thoughts +
GAN-adversarial pipeline, but any stage can be replaced without touching the
engine.

Main Components:
- Engine: Runs the search control flow over pluggable stages
- Generator / Evaluator / Selector / Synthesizer: the stage protocols
- LlmGenerator, AdversarialEvaluator/SinglePassEvaluator, BeamSearch/GreedySearch,
  LlmSynthesizer: the default implementations
- ThoughtData / EvaluationResult: data models

Example:
    from dialectic import create_engine

    engine = create_engine("Your problem here")
    result = await engine.run()
"""

from pathlib import Path

from dotenv import load_dotenv

# Load .env file with override=True so .env values take priority over
# existing environment variables (e.g., GOOGLE_API_KEY).
_env_file = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_file, override=True)

from .agent import (
    Engine,
    build_default_components,
    create_coordinator,
    create_engine,
    run_tot_workflow,
)
from .agent_factory import ROLE_TEMPLATES, create_agent
from .coordinator import Coordinator
from .gan_evaluator import AdversarialEvaluator, SinglePassEvaluator
from .generation import LlmGenerator
from .models import DiscriminatorVerdict, EvaluationResult, ThoughtData
from .protocols import Evaluator, Generator, Selector, Synthesizer
from .selection import BeamSearch, GreedySearch
from .synthesis import LlmSynthesizer

__all__ = [
    # Main entry points
    "create_engine",
    "Engine",
    "build_default_components",
    "run_tot_workflow",
    # Backward-compatible aliases
    "create_coordinator",
    "Coordinator",
    # Stage protocols (the pluggable interfaces)
    "Generator",
    "Evaluator",
    "Selector",
    "Synthesizer",
    # Default stage implementations
    "LlmGenerator",
    "AdversarialEvaluator",
    "SinglePassEvaluator",
    "BeamSearch",
    "GreedySearch",
    "LlmSynthesizer",
    # Data models
    "ThoughtData",
    "EvaluationResult",
    "DiscriminatorVerdict",
    # Agent creation
    "create_agent",
    "ROLE_TEMPLATES",
]

__version__ = "0.3.0"
