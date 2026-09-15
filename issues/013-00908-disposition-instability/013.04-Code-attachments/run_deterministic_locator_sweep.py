"""Run the frozen 009.08 sweep under the 013.04 deterministic contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


EXPECTED_SOURCE_COMMIT = "7a37f58444901c32a6e750e617d8d4d82b9f6202"
EXPECTED_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}
EXPECTED_VERSIONS = {
    "python": (3, 12, 11),
    "torch": "2.14.0",
    "numpy": "2.5.1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--normalized-output", required=True, type=Path)
    args = parser.parse_args()

    source_checkout = args.source_checkout.resolve()
    observed_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_checkout, text=True
    ).strip()
    if observed_commit != EXPECTED_SOURCE_COMMIT:
        raise SystemExit(
            f"source commit mismatch: {observed_commit} != {EXPECTED_SOURCE_COMMIT}"
        )
    observed_env = {name: os.environ.get(name) for name in EXPECTED_ENV}
    if observed_env != EXPECTED_ENV:
        raise SystemExit(f"environment mismatch: {observed_env} != {EXPECTED_ENV}")

    import numpy as np
    import torch

    if sys.version_info[:3] != EXPECTED_VERSIONS["python"]:
        raise SystemExit(f"Python mismatch: {sys.version_info[:3]}")
    if torch.__version__ != EXPECTED_VERSIONS["torch"]:
        raise SystemExit(f"PyTorch mismatch: {torch.__version__}")
    if np.__version__ != EXPECTED_VERSIONS["numpy"]:
        raise SystemExit(f"NumPy mismatch: {np.__version__}")
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        raise SystemExit("contract requires CPU with CUDA and MPS unavailable")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    module_dir = source_checkout / "experiments" / "sfp_representation"
    sys.path.insert(0, str(module_dir))
    sweep = importlib.import_module("locator_sweep")
    if Path(sweep.__file__).resolve() != module_dir / "locator_sweep.py":
        raise SystemExit(f"wrong locator_sweep import: {sweep.__file__}")

    result = sweep.audit(sweep.configurations(), verbose=True)
    if not result["verdict"]["agrees"]:
        raise SystemExit(result["verdict"]["statement"])

    raw = sweep.render(result)
    normalized = sweep.render(sweep.strip_non_replayable(json.loads(raw)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(raw)
    args.normalized_output.write_text(normalized)
    sweep._report(result)
    print(f"raw_sha256={sha256(args.output)}", flush=True)
    print(f"normalized_sha256={sha256(args.normalized_output)}", flush=True)
    print(f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}")
    print(f"intraop_threads={torch.get_num_threads()}")
    print(f"interop_threads={torch.get_num_interop_threads()}")
    print(f"platform={platform.platform()}")


if __name__ == "__main__":
    main()
