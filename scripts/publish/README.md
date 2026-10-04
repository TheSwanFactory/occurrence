# Publishing the papers

The two papers are mirrored in three places, which must stay identical:

| Paper | GitHub (`main`) | `occurrence/theory` package | ihack.us page |
|---|---|---|---|
| OT-I | `occurrence-theory.md` | `papers/occurrence-theory.md` | [1798](https://ihack.us/occurrence-theory/) |
| OT-II | `occurrence-theory-ii.md` | `papers/occurrence-theory-ii.md` | [1872](https://ihack.us/occurrence-theory/ot-ii-born-channel/) |

## Procedure

1. **Edit the package copy** in the local QuiltSync folder
   (`~/QuiltSync/occurrence/theory/papers/`). QuiltSync publishes it; do not
   push to the bucket directly.
2. **Copy the same bytes to the repo root**, run the paper audits
   (`uv run python verify/occurrence_i_audit.py`,
   `uv run python verify/occurrence_ii_audit.py`, `uv run python -m pytest verify/`),
   and merge. Confirm `shasum -a 256` matches across both copies.
3. **Render WordPress HTML** from `main`:

   ```sh
   uv run scripts/publish/md2wp.py occurrence-theory.md -o /tmp/ot-i.html
   uv run scripts/publish/md2wp.py occurrence-theory-ii.md -o /tmp/ot-ii.html
   ```

4. **Update the pages** with the rendered HTML as the full `content`
   (WordPress.com MCP `pages.update` on site 994203). Pages are live as soon as
   they are saved.
5. **Verify** the published text and the edge caches:

   ```sh
   uv run scripts/publish/compare_rendered.py /tmp/ot-i.html 1798
   uv run scripts/publish/compare_rendered.py /tmp/ot-ii.html 1872
   ```

   Then fetch each page URL with and without `Cache-Control: no-cache` and check
   for the new version line.

## Checking the renderer

Before trusting a change to `md2wp.py`, render the paper version that produced
the current live page and run `compare_rendered.py` against it: the result
should be `identical`. On 2026-10-04 the renderer reproduced page 1872's stored
content byte-for-byte from the 2026-08-24 paper.
