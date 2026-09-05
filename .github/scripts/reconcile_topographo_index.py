#!/usr/bin/env python3
"""Reconcile an interrupted topographo upload with a Python package index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

PACKAGE_NAME = "topographo"


class ReconciliationError(RuntimeError):
    """The remote release cannot be safely reconciled."""


def fail(message: str) -> None:
    print(f"::error::{message}", file=sys.stderr)
    raise ReconciliationError(message)


def local_artifacts(dist_dir: Path) -> dict[str, tuple[Path, str]]:
    paths = [*sorted(dist_dir.glob("*.whl")), *sorted(dist_dir.glob("*.tar.gz"))]
    wheels = [path for path in paths if path.suffix == ".whl"]
    sdists = [path for path in paths if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1:
        fail(
            f"Expected exactly one wheel and one sdist in {dist_dir}; found "
            f"{len(wheels)} wheel(s) and {len(sdists)} sdist(s)."
        )
    return {
        path.name: (path, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in paths
    }


def fetch_release(base_url: str, version: str) -> dict | None:
    endpoint = f"{base_url.rstrip('/')}/{quote(PACKAGE_NAME)}/{quote(version)}/json"
    request = Request(endpoint, headers={"User-Agent": "topographo-release-reconcile/1"})
    try:
        with urlopen(request, timeout=30) as response:
            if response.status != 200:
                fail(f"Package index returned unexpected HTTP {response.status} for {endpoint}.")
            return json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            return None
        fail(f"Package index returned HTTP {exc.code} for {endpoint}.")
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        fail(f"Package-index reconciliation failed closed for {endpoint}: {exc}")


def compare_release(
    release: dict | None, expected: dict[str, tuple[Path, str]]
) -> list[str] | None:
    if release is None:
        return None

    remote: dict[str, str] = {}
    for item in release.get("urls", []):
        filename = item.get("filename")
        digest = item.get("digests", {}).get("sha256")
        if not isinstance(filename, str) or not isinstance(digest, str):
            fail("Package-index response contains an artifact without filename/SHA-256.")
        if filename in remote:
            fail(f"Package index returned duplicate artifact metadata for {filename}.")
        remote[filename] = digest

    unexpected = sorted(set(remote) - set(expected))
    if unexpected:
        fail(f"Package index contains unexpected artifacts: {', '.join(unexpected)}.")
    for filename, digest in remote.items():
        expected_digest = expected[filename][1]
        if digest != expected_digest:
            fail(
                f"Remote SHA-256 for {filename} is {digest}, expected {expected_digest}."
            )
    return sorted(set(expected) - set(remote))


def wait_for_state(
    base_url: str,
    version: str,
    expected: dict[str, tuple[Path, str]],
    attempts: int,
    delay: float,
    require_complete: bool,
) -> list[str]:
    last_missing: list[str] | None = None
    for attempt in range(1, attempts + 1):
        missing = compare_release(fetch_release(base_url, version), expected)
        if missing is not None and (not require_complete or not missing):
            return missing
        last_missing = missing
        if attempt < attempts:
            state = "not visible" if missing is None else f"missing {', '.join(missing)}"
            print(f"Index state is {state}; retrying ({attempt}/{attempts})...")
            time.sleep(delay)
    if last_missing is None:
        fail("No release artifacts became visible after the interrupted upload.")
    fail(f"Package index is still missing artifacts: {', '.join(last_missing)}.")


def write_outputs(path: str | None, values: dict[str, str]) -> None:
    for key, value in values.items():
        print(f"{key}={value}")
    if path:
        with Path(path).open("a") as output:
            for key, value in values.items():
                output.write(f"{key}={value}\n")


def command_prepare(args: argparse.Namespace) -> None:
    expected = local_artifacts(Path(args.dist_dir))
    missing = wait_for_state(
        args.base_url,
        args.version,
        expected,
        args.attempts,
        args.delay,
        require_complete=False,
    )
    missing_dir = Path(args.missing_dir)
    missing_dir.mkdir(parents=True, exist_ok=True)
    for filename in missing:
        shutil.copy2(expected[filename][0], missing_dir / filename)
        print(f"Prepared missing artifact: {filename}")
    write_outputs(
        args.github_output,
        {
            "index_complete": str(not missing).lower(),
            "has_missing": str(bool(missing)).lower(),
        },
    )


def command_verify(args: argparse.Namespace) -> None:
    expected = local_artifacts(Path(args.dist_dir))
    wait_for_state(
        args.base_url,
        args.version,
        expected,
        args.attempts,
        args.delay,
        require_complete=True,
    )
    print(f"Package index exactly matches the tested {PACKAGE_NAME} {args.version} artifacts.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, handler in (("prepare", command_prepare), ("verify", command_verify)):
        command = subparsers.add_parser(name)
        command.add_argument("--base-url", required=True)
        command.add_argument("--version", required=True)
        command.add_argument("--dist-dir", default="dist")
        command.add_argument("--attempts", type=int, default=6)
        command.add_argument("--delay", type=float, default=10)
        command.set_defaults(handler=handler)
        if name == "prepare":
            command.add_argument("--missing-dir", default="missing-dist")
            command.add_argument("--github-output")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.attempts < 1 or args.delay < 0:
        print("::error::attempts must be positive and delay must be non-negative.", file=sys.stderr)
        return 1
    try:
        args.handler(args)
    except ReconciliationError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
