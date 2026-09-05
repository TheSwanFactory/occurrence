#!/usr/bin/env python3
"""Validate and support topographo releases from GitHub Actions."""

from __future__ import annotations

import argparse
import email.parser
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tomllib
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile

PACKAGE_NAME = "topographo"
PROJECT_FILE = Path("pyproject.toml")
LOCK_FILE = Path("uv.lock")
CHANGELOG_FILE = Path("CHANGELOG.md")
ZERO_SHA = "0" * 40
TEST_CONFIRMATION = "publish-to-testpypi"


class ReleaseError(RuntimeError):
    """A release invariant was not satisfied."""


def fail(message: str) -> None:
    print(f"::error::{message}", file=sys.stderr)
    raise ReleaseError(message)


def load_project_version_text(data: bytes | str, label: str) -> str:
    try:
        parsed = tomllib.loads(data.decode() if isinstance(data, bytes) else data)
        version = parsed["project"]["version"]
    except (KeyError, TypeError, tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        fail(f"Cannot read [project].version from {label}: {exc}")
    if not isinstance(version, str) or not version:
        fail(f"[project].version in {label} must be a non-empty string.")
    return version


def parse_canonical_version(raw: str, label: str):
    from packaging.version import InvalidVersion, Version

    try:
        parsed = Version(raw)
    except InvalidVersion as exc:
        fail(f"{label} is not a valid PEP 440 version: {raw!r} ({exc})")
    if str(parsed) != raw:
        fail(
            f"{label} must use canonical PEP 440 spelling: "
            f"{raw!r} normalizes to {str(parsed)!r}."
        )
    if parsed.local is not None:
        fail(f"{label} must not contain a local version segment: {raw!r}.")
    return parsed


def git_file(ref: str, path: Path) -> bytes:
    if not ref or ref == ZERO_SHA:
        fail(f"Cannot compare release metadata: invalid Git reference {ref!r}.")
    result = subprocess.run(
        ["git", "show", f"{ref}:{path.as_posix()}"],
        check=False,
        capture_output=True,
    )
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        fail(f"Cannot read {path} at {ref}: {detail}")
    return result.stdout


def validate_lockfile(version: str) -> None:
    try:
        lock = tomllib.loads(LOCK_FILE.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        fail(f"Cannot parse {LOCK_FILE}: {exc}")
    matches = [
        package
        for package in lock.get("package", [])
        if package.get("name") == PACKAGE_NAME
        and package.get("source", {}).get("editable") == "."
    ]
    if len(matches) != 1:
        fail(
            f"{LOCK_FILE} must contain exactly one editable {PACKAGE_NAME!r} "
            f"package; found {len(matches)}."
        )
    locked_version = matches[0].get("version")
    if locked_version != version:
        fail(
            f"Editable {PACKAGE_NAME} version in {LOCK_FILE} is "
            f"{locked_version!r}, expected {version!r}."
        )


def validate_changelog(version: str) -> None:
    try:
        changelog = CHANGELOG_FILE.read_text()
    except OSError as exc:
        fail(f"Cannot read {CHANGELOG_FILE}: {exc}")
    heading = re.compile(
        rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$", re.MULTILINE
    )
    if not heading.search(changelog):
        fail(
            f"{CHANGELOG_FILE} must contain a dated release heading exactly like "
            f"'## [{version}] - YYYY-MM-DD'."
        )


def write_outputs(path: str | None, values: dict[str, str]) -> None:
    for key, value in values.items():
        print(f"{key}={value}")
    if path:
        with Path(path).open("a") as output:
            for key, value in values.items():
                output.write(f"{key}={value}\n")


def validate_forward_change(current_raw: str, current, prior_ref: str, label: str) -> bool:
    prior_raw = load_project_version_text(git_file(prior_ref, PROJECT_FILE), prior_ref)
    prior = parse_canonical_version(prior_raw, f"Previous version at {prior_ref}")
    if current_raw == prior_raw:
        print(f"{label}: project version is unchanged at {current_raw}; no release requested.")
        return False
    if current <= prior:
        fail(
            f"{label}: project version must move forward from {prior_raw} to a new "
            f"PEP 440 version, not {current_raw}."
        )
    validate_changelog(current_raw)
    print(f"{label}: validated forward version change {prior_raw} -> {current_raw}.")
    return True


def command_validate_source(args: argparse.Namespace) -> None:
    current_raw = load_project_version_text(PROJECT_FILE.read_bytes(), str(PROJECT_FILE))
    current = parse_canonical_version(current_raw, "Current project version")
    validate_lockfile(current_raw)

    production_release = False
    test_release = False
    event = args.event_name

    if event == "push" and args.ref == "refs/heads/main":
        production_release = validate_forward_change(
            current_raw, current, args.before, "main push"
        )
    elif event == "pull_request":
        validate_forward_change(current_raw, current, args.base_sha, "pull request")
    elif event == "workflow_dispatch" and args.publish_prerelease:
        if args.confirmation != TEST_CONFIRMATION:
            fail(
                "TestPyPI publication requires confirmation exactly equal to "
                f"{TEST_CONFIRMATION!r}."
            )
        if args.expected_version != current_raw:
            fail(
                f"Expected version {args.expected_version!r} does not match the "
                f"committed project version {current_raw!r}."
            )
        parse_canonical_version(args.expected_version, "Expected prerelease version")
        if current.pre is None:
            fail(
                "Manual publication is restricted to PEP 440 prereleases with an "
                "a, b, or rc segment and can target TestPyPI only."
            )
        changed = validate_forward_change(
            current_raw, current, args.baseline_ref, "TestPyPI dispatch"
        )
        if not changed:
            fail("TestPyPI dispatch version must be newer than the default branch.")
        test_release = True
    else:
        print(f"{event}: validation/build only; no package publication requested.")

    write_outputs(
        args.github_output,
        {
            "version": current_raw,
            "production_release": str(production_release).lower(),
            "test_release": str(test_release).lower(),
            "prerelease": str(current.is_prerelease).lower(),
        },
    )


def distribution_metadata(path: Path) -> tuple[str, str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                fail(f"{path} must contain exactly one .dist-info/METADATA file.")
            payload = archive.read(names[0]).decode("utf-8")
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO")
            ]
            if len(members) != 1:
                fail(f"{path} must contain exactly one top-level PKG-INFO file.")
            extracted = archive.extractfile(members[0])
            if extracted is None:
                fail(f"Cannot read PKG-INFO from {path}.")
            payload = extracted.read().decode("utf-8")
    else:
        fail(f"Unsupported distribution artifact: {path}")

    metadata = email.parser.Parser().parsestr(payload)
    return metadata.get("Name", ""), metadata.get("Version", "")


def command_verify_dist(args: argparse.Namespace) -> None:
    dist_dir = Path(args.dist_dir)
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        fail(
            f"Expected exactly one wheel and one source distribution in {dist_dir}; "
            f"found {len(wheels)} wheel(s) and {len(sdists)} sdist(s)."
        )
    for artifact in [*wheels, *sdists]:
        name, version = distribution_metadata(artifact)
        if name != PACKAGE_NAME or version != args.version:
            fail(
                f"{artifact} metadata is {name} {version}, expected "
                f"{PACKAGE_NAME} {args.version}."
            )
        print(f"Verified {artifact}: {name} {version}")


def command_check_index(args: argparse.Namespace) -> None:
    endpoint = f"{args.base_url.rstrip('/')}/{quote(PACKAGE_NAME)}/{quote(args.version)}/json"
    request = Request(endpoint, headers={"User-Agent": "topographo-release-check/1"})
    try:
        with urlopen(request, timeout=30) as response:
            status = response.status
    except HTTPError as exc:
        if exc.code == 404:
            print(f"{PACKAGE_NAME} {args.version} is not present at {args.base_url}.")
            return
        fail(f"Package-index preflight returned HTTP {exc.code} for {endpoint}.")
    except (URLError, TimeoutError) as exc:
        fail(f"Package-index preflight failed closed for {endpoint}: {exc}")
    if status == 200:
        fail(f"{PACKAGE_NAME} {args.version} already exists at {args.base_url}.")
    fail(f"Package-index preflight returned unexpected HTTP {status} for {endpoint}.")


def remote_tag_target(remote: str, tag: str) -> str | None:
    direct_ref = f"refs/tags/{tag}"
    peeled_ref = f"{direct_ref}^{{}}"
    result = subprocess.run(
        ["git", "ls-remote", "--tags", remote, direct_ref, peeled_ref],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        fail(f"Cannot inspect remote tag {tag}: {result.stderr.strip()}")
    refs = {}
    for line in result.stdout.splitlines():
        sha, ref = line.split(maxsplit=1)
        refs[ref] = sha
    if not refs:
        return None
    return refs.get(peeled_ref, refs.get(direct_ref))


def command_check_tag(args: argparse.Namespace) -> None:
    tag = f"v{args.version}"
    target = remote_tag_target(args.remote, tag)
    if target is not None and target != args.commit:
        fail(
            f"Existing remote tag {tag} points to {target}, not release commit "
            f"{args.commit}."
        )
    exists = target is not None
    print(
        f"Remote tag {tag} "
        + (f"already points to {target}." if exists else "does not exist yet.")
    )
    write_outputs(args.github_output, {"tag": tag, "tag_exists": str(exists).lower()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-source")
    validate.add_argument("--event-name", required=True)
    validate.add_argument("--ref", default="")
    validate.add_argument("--before", default="")
    validate.add_argument("--base-sha", default="")
    validate.add_argument("--baseline-ref", default="refs/remotes/origin/main")
    validate.add_argument("--publish-prerelease", action="store_true")
    validate.add_argument("--expected-version", default="")
    validate.add_argument("--confirmation", default="")
    validate.add_argument("--github-output")
    validate.set_defaults(handler=command_validate_source)

    verify_dist = subparsers.add_parser("verify-dist")
    verify_dist.add_argument("--dist-dir", default="dist")
    verify_dist.add_argument("--version", required=True)
    verify_dist.set_defaults(handler=command_verify_dist)

    check_index = subparsers.add_parser("check-index")
    check_index.add_argument("--base-url", required=True)
    check_index.add_argument("--version", required=True)
    check_index.set_defaults(handler=command_check_index)

    check_tag = subparsers.add_parser("check-tag")
    check_tag.add_argument("--remote", default="origin")
    check_tag.add_argument("--version", required=True)
    check_tag.add_argument("--commit", required=True)
    check_tag.add_argument("--github-output")
    check_tag.set_defaults(handler=command_check_tag)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.handler(args)
    except ReleaseError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
