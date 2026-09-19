"""Apply the frozen 009.08 analysis unchanged to a supplied sweep."""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


EXPECTED_SOURCE_COMMIT = "7a37f58444901c32a6e750e617d8d4d82b9f6202"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--sweep", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source_checkout = args.source_checkout.resolve()
    observed_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_checkout, text=True
    ).strip()
    if observed_commit != EXPECTED_SOURCE_COMMIT:
        raise SystemExit(f"analysis source mismatch: {observed_commit}")
    module_dir = source_checkout / "experiments" / "sfp_representation"
    sys.path.insert(0, str(module_dir))
    analysis = importlib.import_module("locator_analysis")
    analysis.SWEEP = args.sweep.resolve()
    result = analysis.analyse()
    rendered = analysis.render(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    analysis._report(result)
    print(f"analysis_agrees={result['verdict']['agrees']}", flush=True)
    print(f"analysis_statement={result['verdict']['statement']}", flush=True)


if __name__ == "__main__":
    main()
