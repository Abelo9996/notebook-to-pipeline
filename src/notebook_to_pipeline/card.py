"""A small, shareable result card: one SVG and one PNG summarising a pipeline run.

The SVG is rendered to PNG with resvg-py, a self-contained Rust renderer shipped as a pure wheel
(no cairo or other system libraries). Text uses whatever sans-serif and monospace fonts the machine
has; the card is a share image, not a pixel-compared artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

BG = "#0b1020"
PANEL = "#111736"
BORDER = "#242c52"
TEXT = "#e8ecf5"
MUTED = "#9aa4bf"
ACCENT = "#6aa6ff"
GOOD = "#2ea043"
BAD = "#e5534b"
WARN = "#d9a441"

WIDTH = 1200
HEIGHT = 630


def _esc(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _ellipsize(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _rounded(x: int, y: int, w: int, h: int, r: int, fill: str, stroke: str | None = None) -> str:
    s = f' stroke="{stroke}" stroke-width="2"' if stroke else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" fill="{fill}"{s}/>'


def build_svg(data: dict[str, Any]) -> str:
    """Build the card SVG from a small data dict.

    Keys: notebook (file name), verdict ("equivalent" or anything else), locked (int, outputs
    locked), locked_detail (str, e.g. "26 variables, 3 figures"), findings (int, hidden-state and
    related findings caught), command (the check-yours command), refactored (bool)."""
    notebook = _ellipsize(str(data.get("notebook", "notebook.ipynb")), 42)
    verdict = str(data.get("verdict", "")).lower()
    equivalent = verdict == "equivalent"
    locked = int(data.get("locked", 0))
    locked_detail = str(data.get("locked_detail", "")).strip()
    findings = int(data.get("findings", 0))
    command = _ellipsize(str(data.get("command", "uvx notebook-to-pipeline pipeline x.ipynb")), 54)
    refactored = bool(data.get("refactored", False))

    badge_text = "EQUIVALENT" if equivalent else "NOT EQUIVALENT"
    badge_fill = GOOD if equivalent else BAD
    badge_w = 46 + len(badge_text) * 22

    outputs_word = "output" if locked == 1 else "outputs"
    lock_line = f"{locked} {outputs_word} locked"
    if locked_detail:
        lock_line += f"  ({locked_detail})"

    findings_word = "issue" if findings == 1 else "issues"
    findings_line = f"{findings} hidden-state {findings_word} caught"
    if findings == 0:
        findings_line = "no hidden-state issues found"

    sub = (
        "refactored into stages, outputs re-verified" if refactored else "outputs locked by a test"
    )

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="Helvetica, Arial, sans-serif">'
    )
    parts.append(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BG}"/>')
    parts.append(_rounded(40, 40, WIDTH - 80, HEIGHT - 80, 28, PANEL, BORDER))

    # Header
    parts.append(
        f'<text x="88" y="130" fill="{ACCENT}" font-size="30" font-weight="700" '
        f'letter-spacing="1">notebook-to-pipeline</text>'
    )

    # Title: notebook -> pipeline
    parts.append(
        f'<text x="88" y="232" fill="{TEXT}" font-size="58" font-weight="800">'
        f'{_esc(notebook)} <tspan fill="{MUTED}">→</tspan> pipeline</text>'
    )

    # Badge
    by = 290
    parts.append(_rounded(88, by, badge_w, 72, 16, badge_fill))
    parts.append(
        f'<text x="{88 + badge_w // 2}" y="{by + 48}" fill="#ffffff" font-size="34" '
        f'font-weight="800" text-anchor="middle">{_esc(badge_text)}</text>'
    )
    parts.append(
        f'<text x="{88 + badge_w + 28}" y="{by + 47}" fill="{MUTED}" font-size="28">'
        f"{_esc(sub)}</text>"
    )

    # Stats
    parts.append(
        f'<text x="88" y="446" fill="{TEXT}" font-size="34" font-weight="600">'
        f"{_esc(lock_line)}</text>"
    )
    parts.append(
        f'<text x="88" y="498" fill="{WARN if findings else MUTED}" font-size="34" '
        f'font-weight="600">{_esc(findings_line)}</text>'
    )

    # Footer command
    fy = 536
    parts.append(_rounded(88, fy, WIDTH - 176, 56, 12, BG, BORDER))
    parts.append(
        f'<text x="112" y="{fy + 37}" fill="{MUTED}" font-size="26" '
        f'font-family="Menlo, Consolas, monospace">check yours: '
        f'<tspan fill="{ACCENT}">{_esc(command)}</tspan></text>'
    )

    parts.append("</svg>")
    return "".join(parts)


def render_png(svg: str) -> bytes:
    """Render the SVG to PNG bytes with resvg-py."""
    import resvg_py

    out = resvg_py.svg_to_bytes(svg_string=svg)
    return bytes(out)


def write_card(out_dir: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    """Write card.svg and card.png into out_dir. Returns the paths and whether the PNG was made."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    svg = build_svg(data)
    svg_path = out / "card.svg"
    svg_path.write_text(svg, encoding="utf-8")
    result: dict[str, Any] = {"svg": str(svg_path), "png": None}
    try:
        png = render_png(svg)
        png_path = out / "card.png"
        png_path.write_bytes(png)
        result["png"] = str(png_path)
    except Exception as exc:  # resvg-py missing or a render error: the SVG is still written
        result["png_error"] = f"{type(exc).__name__}: {exc}"
    return result
