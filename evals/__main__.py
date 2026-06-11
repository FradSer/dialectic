"""CLI entry point: ``uv run python -m evals``.

As a development tool (unlike the library), this loads ``dialectica/.env``.
"""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from dialectica import create_engine

from .baseline import SingleCallBaseline, create_baseline_agent
from .harness import render_markdown, run_eval
from .judge import BlindJudge, create_judge_agent
from .problems import DEFAULT_PROBLEMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evals", description="Compare the engine against a single-call baseline."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=len(DEFAULT_PROBLEMS),
        help="Number of benchmark problems to run.",
    )
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--gan-rounds", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=7.0)
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also write the full report as JSON to this path.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    load_dotenv(Path(__file__).resolve().parent.parent / "dialectica" / ".env")

    def engine_factory(statement: str):
        return create_engine(
            statement,
            max_depth=args.max_depth,
            beam_width=args.beam_width,
            max_gan_rounds=args.gan_rounds,
            score_threshold=args.threshold,
        )

    report = await run_eval(
        DEFAULT_PROBLEMS[: args.limit],
        engine_factory=engine_factory,
        baseline=SingleCallBaseline(create_baseline_agent()),
        judge=BlindJudge(create_judge_agent()),
    )

    print(render_markdown(report))
    if args.json:
        args.json.write_text(json.dumps(report.model_dump(), indent=2))
        print(f"JSON report written to {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
