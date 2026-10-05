# /// script
# requires-python = ">=3.11"
# ///
"""Check that a live ihack.us page shows the text of a rendered paper.

Fetches the page's public ``content.rendered`` from the WordPress.com REST API
(cache-busted) and compares it with the local ``md2wp.py`` output after
stripping tags, decoding entities, undoing wptexturize's smart punctuation, and
collapsing whitespace. Exits 0 when the normalized texts are identical.

Usage::

    uv run scripts/publish/compare_rendered.py ot-i.html 1798
    uv run scripts/publish/compare_rendered.py ot-ii.html 1872
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import re
import time
import urllib.request
from pathlib import Path

SITE_ID = 994203  # ihack.us
API = "https://public-api.wordpress.com/wp/v2/sites/{site}/pages/{page}?_fields=content,modified_gmt&cb={cb}"

# wptexturize rewrites ASCII punctuation; map it back before comparing.
TEXTURIZE = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    " ": " ",
}


def normalize(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    for a, b in TEXTURIZE.items():
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def fetch(page: int, site: int = SITE_ID) -> tuple[str, str]:
    url = API.format(site=site, page=page, cb=time.time_ns())
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
    return data["content"]["rendered"], data.get("modified_gmt", "?")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rendered", type=Path, help="md2wp.py output")
    parser.add_argument("page", type=int, help="WordPress page ID (1798 or 1872)")
    parser.add_argument("--site", type=int, default=SITE_ID)
    args = parser.parse_args(argv)

    live, modified = fetch(args.page, args.site)
    a = normalize(args.rendered.read_text(encoding="utf-8"))
    b = normalize(live)
    print(f"page {args.page} modified_gmt={modified} local={len(a)} live={len(b)}")
    if a == b:
        print("identical")
        return 0

    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    diffs = [op for op in sm.get_opcodes() if op[0] != "equal"]
    for tag, i1, i2, j1, j2 in diffs[:12]:
        print(
            f"{tag}: {a[max(0, i1 - 40) : i2 + 40]!r}\n   => {b[max(0, j1 - 40) : j2 + 40]!r}"
        )
    print(f"DIFFER: {len(diffs)} change(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
