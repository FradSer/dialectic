# Dialectica ![](https://img.shields.io/badge/A%20FRAD%20PRODUCT-WIP-yellow)

[![PyPI](https://img.shields.io/pypi/v/dialectica.svg)](https://pypi.org/project/dialectica/) [![Twitter Follow](https://img.shields.io/twitter/follow/FradSer?style=social)](https://twitter.com/FradSer) [![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![Framework](https://img.shields.io/badge/Framework-ADK%202.1-orange.svg)](https://google.github.io/adk-docs/) [![Evaluation](https://img.shields.io/badge/Evaluation-GAN%20Adversarial-purple.svg)]()

English | [简体中文](https://github.com/FradSer/dialectica/blob/main/README.zh-CN.md)

**Dialectica** is a pluggable adversarial reasoning engine. It searches a tree of "thoughts" where each thought is generated, adversarially evaluated and iteratively refined, then synthesized into an answer — *thesis → antithesis → synthesis* (Generator → Discriminator → Synthesizer). Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch)'s propose→evaluate→keep-best loop and Claude Code's composable workflows, every stage is a swappable component; the default wiring is Tree-of-Thoughts + a GAN-style evaluation loop on Google ADK 2.1.

## Install

Use it as a library in your own project:

```bash
uv add dialectica
# or: pip install dialectica
```

```python
import os, asyncio
from dialectica import create_engine

os.environ["GOOGLE_API_KEY"] = "..."          # the app owns env setup

async def main():
    result = await create_engine("Your problem here").run()
    print(result["final_answer"])

asyncio.run(main())
```

The library reads configuration from `os.environ` and does **not** load `.env`
itself. To work on Dialectica instead, see [Development](#development).

## Key Features

### 🧩 Pluggable engine (thesis → antithesis → synthesis)
The `Engine` owns only the search *control flow*; every decision is delegated to
an injected component, so any stage can be swapped without touching the engine:

| Stage | Role | Default |
|-------|------|---------|
| `Generator` | propose thoughts (**thesis**) | `LlmGenerator` |
| `Evaluator` | critique & refine (**antithesis**) | `AdversarialEvaluator` |
| `Selector` | choose the frontier | `BeamSearch` |
| `Synthesizer` | combine into an answer (**synthesis**) | `LlmSynthesizer` |

Retarget it at code review, research, or decision-making just by changing the
generator's prompts or swapping a stage — see [Pluggable Architecture](#pluggable-architecture).

### 🔄 GAN-style adversarial evaluation (keep-best)
Each thought undergoes **iterative adversarial refinement** rather than a single pass:
1. **Discriminator** scores it with a structured verdict (score, flaws, suggestions)
2. **Generator** refines it from that critique
3. **Discriminator** re-scores
4. Loop until the quality threshold, a terminate signal, or `max_gan_rounds`

Refinement is **not assumed monotonic** — the loop keeps the *best-scoring* round
(à la autoresearch's "keep only what beats the current best"), and the node stores
that refined text so synthesis works on the improved version, not the original.

### 🌳 Tree search with merit-based beam
- **Strategies are scored before the beam** — the frontier reflects merit, not generation order
- **Beam search** keeps the top-k most promising paths (`BeamSearch`, or `GreedySearch`)
- **Pruning**: paths below threshold are dropped; exploration stops when the beam empties
- **Multi-node synthesis**: the final answer integrates the top scoring thoughts across branches

### 📊 Structured evaluation results
The `Discriminator` returns a `DiscriminatorVerdict` via ADK `output_schema` (no
fragile text parsing). The engine wraps it as an `EvaluationResult`:
`score`, `flaws`, `suggestions`, `should_terminate`, `reasoning`,
`adversarial_rounds`, `refined_thought`, and the full per-round `history`.

## Architecture

```
User Problem
    ↓
Engine — Phase 1: Initialize
    ↓
Generator expands root → initial strategies
    ↓ (each strategy scored by the Evaluator before it can enter the beam)
Engine — Phase 2: Explore (beam search)
    ↓
For each node in the Selector's frontier:
    ├── Generator expands it into children
    └── for each child, Evaluator runs the GAN loop:
        ├── Discriminator scores (structured verdict)
        ├── Generator refines from the critique
        ├── re-score, keep the best round
        └── persist the refined thought + score on the node
    → children ≥ threshold form the next beam
    ↓
Engine — Phase 3: Synthesize
    ↓
Synthesizer integrates the top thoughts
    ↓
Final Answer (+ thought_tree, best_path, stats)
```

## Workflow Phases

### Phase 1: Initialization
- Creates the root node from the user problem
- `Generator.expand(root)` produces the initial strategies (validated via `ThoughtData`)
- **Each strategy is adversarially scored**, then the ones clearing the threshold seed the beam (falling back to the Selector's top-k if none clear it)

### Phase 2: Exploration (beam search)
Iterates up to `max_depth` times:
1. **Select**: `Selector.select(...)` picks the frontier from the active beam
2. **Generate**: `Generator.expand(parent)` creates child thoughts
3. **Evaluate**: `Evaluator.evaluate(...)` runs the GAN loop, keeping the best round and persisting the refined thought
4. **Filter**: children scoring ≥ `score_threshold` form the next beam

Exploration stops when the beam empties or `max_depth` is reached.

### Phase 3: Synthesis
- `Synthesizer.synthesize(...)` takes the top-scoring evaluated thoughts
- Produces a coherent, comprehensive final answer

## Development

To work on Dialectica itself (not needed just to *use* it — see [Install](#install)):

```bash
git clone https://github.com/FradSer/dialectica
cd dialectica
uv sync
cp dialectica/.env.example dialectica/.env   # add GOOGLE_API_KEY for the live e2e test
```

Then run the suite — see [Testing](#testing).

## Configuration

### Environment Variables

**Model Configuration:**
```bash
# Default model for all agents
DEFAULT_MODEL_CONFIG=google:gemini-3.5-flash

# Role-specific overrides (optional)
GENERATOR_MODEL_CONFIG=google:gemini-3.1-pro-preview
DISCRIMINATOR_MODEL_CONFIG=google:gemini-3.1-pro-preview
SYNTHESIZER_MODEL_CONFIG=google:gemini-3.1-pro-preview
```

**Supported Providers:**
- `google:gemini-3.5-flash` (Google AI Studio)
- `openrouter:anthropic/claude-3.5-sonnet` (OpenRouter)
- `openai:gpt-4o` (OpenAI)

**API Credentials:**
```bash
# Google AI Studio
GOOGLE_API_KEY=your-key-here

# Or Vertex AI
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=your-project
GOOGLE_CLOUD_LOCATION=us-central1

# OpenRouter
OPENROUTER_API_KEY=sk-or-...

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1
```

### Engine Parameters

```python
engine = create_engine(
    problem="Your problem statement",
    max_depth=4,              # Max tree depth
    beam_width=3,             # Active paths per iteration
    max_gan_rounds=3,         # Max adversarial refinement rounds
    score_threshold=7.0,      # Min score to continue
    synthesizer_model=None,   # Optional model override
)
```

## Usage Examples

### Basic Usage

```python
from dialectica import create_engine

# Create the engine
engine = create_engine(
    "Design a sustainable urban transport system"
)

# Run workflow
result = await engine.run()

# Access results
print(result["final_answer"])
print(f"Generated {len(result['thought_tree'])} thoughts")
print(f"Best path: {result['best_path']}")
```

### Inspecting the result

`run()` returns the answer plus the full search trace:

```python
result = await engine.run()
result["final_answer"]   # synthesized answer
result["best_path"]      # node ids from root to the highest-scoring thought
result["thought_tree"]   # every node, with scores and per-round GAN history
result["stats"]          # total_thoughts, max_depth_reached, duration_seconds
```

### Custom Configuration

```python
engine = create_engine(
    problem="Optimize supply chain logistics",
    max_depth=5,
    beam_width=5,
    max_gan_rounds=4,
    score_threshold=8.0,
    synthesizer_model="google:gemini-3.1-pro-preview",
)
```

## Project Structure

```
dialectica/
├── __init__.py           # Public API exports
├── agent.py              # Composition root: create_engine() wires defaults
├── coordinator.py        # Search engine — orchestrates the pluggable stages
├── protocols.py          # Stage interfaces: Generator/Evaluator/Selector/Synthesizer
├── generation.py         # LlmGenerator (default Generator) + list parsing
├── gan_evaluator.py      # AdversarialEvaluator / SinglePassEvaluator (Evaluator)
├── selection.py          # BeamSearch / GreedySearch (Selector)
├── synthesis.py          # LlmSynthesizer (default Synthesizer)
├── agent_runtime.py      # Single LLM-call seam (run_agent)
├── agent_factory.py      # Dynamic agent creation (role templates)
├── models.py             # ThoughtData, DiscriminatorVerdict, EvaluationResult
├── llm_config.py         # Model configuration factory
└── validation.py         # Thought validation utilities
tests/
├── conftest.py           # Loads .env for the e2e skip guard
├── helpers.py            # Deterministic mock LLM stand-ins
├── test_models.py        # Schema / verdict unit tests
├── test_generation.py    # List parsing + generator prompt routing
├── test_gan_evaluator.py # GAN loop + single-pass evaluator (mocked LLM)
├── test_coordinator.py   # Engine control flow (injected fake stages)
├── test_default_pipeline.py  # Default composition integration (mocked LLM)
├── test_eval_harness.py  # Eval harness units (judge normalization, counting, report)
├── test_e2e_live.py      # Real Gemini E2E (marked `e2e`)
└── test_eval_live.py     # Real Gemini eval-harness E2E (marked `e2e`)
evals/                     # Eval harness (dev tool, not shipped in the wheel)
├── problems.py            # Benchmark problems
├── baseline.py            # Single-call baseline (the control arm)
├── judge.py               # Blind pairwise judge with position-swap bias control
├── harness.py             # Orchestration, call counting, report rendering
└── __main__.py            # CLI: uv run python -m evals
```

## Testing

The suite has two tiers:

- **Mocked tests** (default) — fast, deterministic, no API key. They replace
  the LLM call seam with stand-ins and exercise the real orchestration: beam
  search, the GAN refinement loop, pruning, and synthesis.
- **Live E2E** (`@pytest.mark.e2e`) — drives the full workflow against the real
  Gemini API. Deselected by default and auto-skipped when `GOOGLE_API_KEY` is
  unset (loaded from `dialectica/.env`).

```bash
uv run pytest          # mocked tests only (seconds, no key)
uv run pytest -m e2e   # live API E2E (slower, requires GOOGLE_API_KEY)
```

## Evaluation

Does the engine actually beat a single strong-model call? The repo ships an
eval harness (`evals/`, a dev tool — not part of the published package) that
answers this with data instead of vibes:

1. Each benchmark problem (`evals/problems.py`) is solved by the **engine**
   and by a **single-call baseline** (one well-prompted LLM call).
2. A **blind judge** compares the two answers without knowing which is which.
   Position bias is neutralized by judging twice with swapped positions —
   if the two verdicts disagree, the comparison is a tie.
3. The report tallies wins and the cost of each arm (LLM calls, wall time),
   counted through the same `run_agent` seam the tests mock.

```bash
uv run python -m evals                          # all benchmark problems
uv run python -m evals --limit 2 --json out.json
uv run python -m evals --max-depth 3 --beam-width 3 --gan-rounds 2
```

Model overrides via env: `BASELINE_MODEL_CONFIG` and `JUDGE_MODEL_CONFIG`
(same `provider:model_name` format; e.g. point the baseline at
`google:gemini-3.1-pro-preview` to compare against a stronger single call).

### Results (2026-06)

Three runs on the 5 default benchmark problems, all judged blind by
`gemini-3.5-flash` with position-swap bias control (engine config:
`max_depth=2, beam_width=2, max_gan_rounds=2, threshold=7.0`):

| Problem | engine(flash) vs flash | engine(flash) vs **pro** | engine(qwen) vs qwen |
|---------|------------------------|--------------------------|----------------------|
| cloud-costs | tie | engine | engine |
| api-versioning | engine | engine | baseline |
| flaky-tests | engine | engine | engine |
| meeting-overload | tie | baseline | baseline |
| urban-transport | baseline | baseline | tie |
| **Total (W-L-T)** | **2-1-2** | **3-2-0** | **2-2-1** |

flash = `gemini-3.5-flash` · pro = `gemini-3.1-pro-preview` (single call) ·
qwen = `qwen3.6-35b-a3b`. Engine cost: ~20× LLM calls vs the baseline's 1 on
Gemini, ~32× on Qwen (a stricter discriminator triggers more GAN rounds).

What the 15 comparisons (7-5-3 overall) say:

- **The engine reliably wins on technical/engineering problems** — 7-1-1
  across cloud costs, API versioning, and flaky-test remediation. Judges
  repeatedly credited refinement-produced depth: contract-testing pipelines,
  correct HTTP semantics for brownouts (503 vs 410), quarantine engines with
  anti-gaming guardrails.
- **It loses on organizational/social problems** — 0-4-2 on meeting overload
  and urban transport, consistently judged "over-complex, impractical". The
  pattern reproduces across model families, implicating the scaffold (the
  discriminator's criteria) rather than any one model.
- A flash engine beats a single stronger pro call 3-2 — search can buy back
  model-tier quality on technical problems, at ~20× the calls.

Practical takeaway: Dialectica is best used as a **technical-decision
deepener**. The full reports (answers + judge reasoning) land in
`evals/results/` when you run the harness.

## Pluggable Architecture

The `Coordinator` owns only the search *control flow*. Every decision is
delegated to an injected component, so any stage can be swapped without
touching the engine — the engine is a general-purpose reasoning workflow, and
ToT + GAN is just the default wiring.

| Protocol | Responsibility | Default | Alternatives |
|----------|----------------|---------|--------------|
| `Generator` | expand a node into candidate thoughts | `LlmGenerator` | custom prompts/agent |
| `Evaluator` | score (and optionally refine) a thought | `AdversarialEvaluator` (GAN loop) | `SinglePassEvaluator` (cheap) |
| `Selector` | choose the next search frontier | `BeamSearch(width)` | `GreedySearch` |
| `Synthesizer` | combine thoughts into the answer | `LlmSynthesizer` | custom |

`create_engine(...)` wires the defaults. To customize, build the
components yourself and construct `Coordinator` directly:

```python
from dialectica import (
    Coordinator, BeamSearch, SinglePassEvaluator, LlmSynthesizer,
)
from dialectica.agent import build_default_components

# Start from the defaults, then swap a stage:
generator, _evaluator, _selector, synthesizer = build_default_components()
from dialectica.agent_factory import create_agent
from dialectica.models import DiscriminatorVerdict

discriminator = create_agent(
    role="Discriminator", role_name="Discriminator", output_schema=DiscriminatorVerdict
)

engine = Coordinator(
    problem="...",
    generator=generator,
    evaluator=SinglePassEvaluator(discriminator),   # cheaper: no refinement loop
    selector=BeamSearch(width=5),                    # wider frontier
    synthesizer=synthesizer,
    max_depth=3,
    score_threshold=7.0,
)
result = await engine.run()
```

Any object implementing a protocol's method works (they are
`typing.Protocol`, so no subclassing needed) — e.g. a non-LLM heuristic
`Evaluator`, or a `Selector` that keeps a diverse frontier instead of pure
top-k.

## Key Components

### Coordinator
Orchestrates the three-phase workflow against the stage protocols:
- Initialize → Explore → Synthesize
- Manages the thought tree and active beam
- Delegates generation, scoring, selection, and synthesis to injected components

### AgentFactory
Creates agents from role templates:
- Standardized system prompts
- Tool configuration per role
- Model configuration per role
- Runtime agent instantiation

### AdversarialEvaluator
Implements GAN-style evaluation:
- Generator proposes/refines thoughts
- Discriminator critiques with feedback
- Iterative refinement loop
- Structured evaluation results

### ThoughtData Model
Validates thought structure:
- Required fields (id, parent_id, depth, content)
- Optional evaluation data
- GAN round tracking
- Evaluation history

## Performance Considerations

**Token Consumption:**
- GAN evaluation: 2-6 LLM calls per thought (depending on rounds)
- Beam search: beam_width × max_depth iterations
- Typical problem: 50-200 thoughts, 200-800 LLM calls

**Optimization Strategies:**
- Reduce `max_gan_rounds` to 1-2 for faster execution
- Raise `score_threshold` to prune harder; lower it to explore more paths
- Narrow the beam (`beam_width`) or use `GreedySearch` to cut fan-out
- Use a lighter model for the Generator and a stronger one for the Discriminator
- Swap in `SinglePassEvaluator` to skip the refinement loop entirely

## Troubleshooting

### Import Errors
```bash
# Ensure Python 3.11+
python --version

# Reinstall dependencies
rm -rf .venv
uv sync
```

### ADK Version Mismatch
```bash
# Check installed version
uv pip show google-adk

# Should show 2.1.0 or higher
```

### API Key Issues
```bash
# Test Google AI Studio
export GOOGLE_API_KEY=your-key
uv run python -c "from dialectica import create_engine; print('OK')"
```

## Contributing

Contributions welcome! Areas of interest:
- New stage implementations (`Generator` / `Evaluator` / `Selector` / `Synthesizer`)
- Alternative search/selection policies (e.g. diversity-preserving frontiers)
- Performance optimizations
- Documentation improvements
- Test coverage

## License

[MIT](https://github.com/FradSer/dialectica/blob/main/LICENSE)

## References

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — propose → evaluate → keep-best loop
- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Tree of Thoughts Paper](https://arxiv.org/abs/2305.10601)

## Acknowledgments

Built with [Google ADK](https://github.com/google/adk-python), inspired by Tree of Thoughts research, [karpathy/autoresearch](https://github.com/karpathy/autoresearch)'s autonomous keep-best loop, and Claude Code's composable workflows.
