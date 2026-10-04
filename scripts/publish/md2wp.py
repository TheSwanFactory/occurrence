# /// script
# requires-python = ">=3.11"
# dependencies = ["markdown-it-py>=3"]
# ///
"""Render an Occurrence Theory paper as classic WordPress HTML for ihack.us.

The ihack.us pages (1798 = occurrence-theory.md, 1872 = occurrence-theory-ii.md)
store classic, non-block content that WordPress passes through wpautop and a
Markdown filter. This renderer reproduces the stored format exactly; run against
the paper bytes that produced the 2026-08-24 pages, it regenerates page 1872
byte-for-byte.

Rules:
- drop the H1 (the page title carries it);
- escape ``*`` and ``_`` as ``&#042;`` / ``&#095;`` in text and attribute values,
  so the site's Markdown filter cannot reinterpret them;
- unwrap ``<p>`` into blank-line-separated text (wpautop rebuilds paragraphs;
  a single newline would become ``<br>`` and merge paragraphs);
- blank line after block closers, none before ``</blockquote>`` / ``</li>``;
- drop ``class`` attributes from ``<code>``;
- rewrite relative links to the GitHub ``main`` blob URL;
- convert ``$$...$$`` display math to WordPress ``$latex ...$``.

Usage::

    uv run scripts/publish/md2wp.py occurrence-theory.md > ot-i.html
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt

GITHUB_BLOB = "https://github.com/TheSwanFactory/occurrence/blob/main/"
BLOCK_CLOSERS = r"(</h\d>|<hr />|</ol>|</ul>|</blockquote>|</table>|</pre>)"


def _escape(s: str) -> str:
    return s.replace("*", "&#042;").replace("_", "&#095;")


def _escape_markup(html: str) -> str:
    """Escape text nodes and quoted attribute values; leave tag syntax alone."""

    def repl(m: re.Match[str]) -> str:
        tag, text = m.group(1), m.group(2)
        if tag:
            return re.sub(r'"[^"]*"', lambda a: _escape(a.group(0)), tag)
        return _escape(text)

    return re.sub(r"(<[^>]*>)|([^<]+)", repl, html)


def render(markdown: str) -> str:
    src = re.sub(r"\A# [^\n]*\n+", "", markdown)

    # Plain-text sentinels: markdown-it mangles NUL-style placeholders.
    maths: list[str] = []

    def stash(m: re.Match[str]) -> str:
        maths.append("$latex " + m.group(1).strip() + "$")
        return f"XXMATH{len(maths) - 1}XX"

    src = re.sub(r"\$\$(.+?)\$\$", stash, src, flags=re.DOTALL)

    html = MarkdownIt("commonmark").enable("table").render(src)
    html = _escape_markup(html)
    html = re.sub(r"<p>(.*?)</p>\n?", r"\1\n\n", html, flags=re.DOTALL)
    html = re.sub(BLOCK_CLOSERS + r"\n", r"\1\n\n", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    html = re.sub(r"\n\n(</blockquote>|</li>)", r"\n\1", html)
    html = re.sub(r'<code class="[^"]*">', "<code>", html)
    html = re.sub(
        r'href="(?!https?:|#|mailto:)([^"]+)"', rf'href="{GITHUB_BLOB}\1"', html
    )
    html = re.sub(r"XXMATH(\d+)XX", lambda m: maths[int(m.group(1))], html)
    return html.strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paper", type=Path, help="paper Markdown file")
    parser.add_argument(
        "-o", "--output", type=Path, help="write here instead of stdout"
    )
    args = parser.parse_args(argv)

    html = render(args.paper.read_text(encoding="utf-8"))
    if args.output:
        args.output.write_text(html, encoding="utf-8")
    else:
        sys.stdout.write(html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
