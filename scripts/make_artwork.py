"""Write the blochsim logo and its mark into ``docs/_static``.

    python scripts/make_artwork.py

The mark is a Bloch sphere: the magnetization vector, tipped onto the
transverse plane, at the end of the precessing path an excitation traces from
the longitudinal axis. The logo is the mark followed by the wordmark. Each
image is written twice, ``<name>.svg`` for a light background and
``<name>-dark.svg`` for a dark one. The wordmark is SVG text in the reader's
sans-serif font; no font is embedded.
"""

from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "_static"

#: The colours of each theme, by role: the pulserver family's blue and amber.
PALETTE = {
    "light": {
        "ink": "#12212b",
        "muted": "#8a9aa8",
        "blue": "#2b76ad",
        "amber": "#f0a500",
    },
    "dark": {
        "ink": "#e8eef3",
        "muted": "#6f8394",
        "blue": "#5b9bd0",
        "amber": "#ffbe2e",
    },
}

#: Sphere radius, and the elevation the sphere is seen from, in radians.
RADIUS = 46.0
ELEVATION = math.radians(22)

#: Precession turns of the excitation path, and the azimuth it ends at.
TURNS = 2.75
END_AZIMUTH = math.radians(-35)


def _project(x: float, y: float, z: float) -> tuple[float, float, bool]:
    """Screen position of a point on the sphere, and whether it faces the viewer."""
    up = z * math.cos(ELEVATION) + y * math.sin(ELEVATION)
    toward = -y * math.cos(ELEVATION) + z * math.sin(ELEVATION)
    return x, -up, toward >= 0


def _path() -> list[tuple[float, float, bool]]:
    """The tip of the magnetization during excitation: tip angle 0 to 90 degrees."""
    points = []
    n = 400
    for i in range(n + 1):
        t = i / n
        tip = 0.5 * math.pi * t
        azimuth = END_AZIMUTH - 2 * math.pi * TURNS * (1 - t)
        r = RADIUS * math.sin(tip)
        points.append(
            _project(
                r * math.cos(azimuth), r * math.sin(azimuth), RADIUS * math.cos(tip)
            )
        )
    return points


def _runs(points: list[tuple[float, float, bool]]) -> list[tuple[bool, str]]:
    """The path split into runs in front of and behind the sphere's centre."""
    runs: list[tuple[bool, list[tuple[float, float]]]] = []
    for x, y, front in points:
        if not runs or runs[-1][0] != front:
            runs.append((front, [runs[-1][1][-1]] if runs else []))
        runs[-1][1].append((x, y))
    return [
        (front, "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in pts))
        for front, pts in runs
    ]


def _mark(colours: dict[str, str]) -> str:
    """The Bloch sphere, centred on the origin, as SVG elements."""
    ry = RADIUS * math.sin(ELEVATION)
    tip_x, tip_y, _ = _path()[-1]
    length = math.hypot(tip_x, tip_y)
    ux, uy = tip_x / length, tip_y / length
    shaft_x, shaft_y = tip_x - 9 * ux, tip_y - 9 * uy
    head = (
        f"{tip_x:.2f},{tip_y:.2f} "
        f"{shaft_x - 6 * uy:.2f},{shaft_y + 6 * ux:.2f} "
        f"{shaft_x + 6 * uy:.2f},{shaft_y - 6 * ux:.2f}"
    )
    parts = [
        f'<circle r="{RADIUS}" fill="none" stroke="{colours["ink"]}" stroke-width="4"/>',
        f'<path d="M{-RADIUS} 0 A{RADIUS} {ry:.2f} 0 0 1 {RADIUS} 0" fill="none" '
        f'stroke="{colours["muted"]}" stroke-width="2" stroke-dasharray="4 4"/>',
        f'<path d="M{-RADIUS} 0 A{RADIUS} {ry:.2f} 0 0 0 {RADIUS} 0" fill="none" '
        f'stroke="{colours["muted"]}" stroke-width="2"/>',
        f'<line x1="0" y1="0" x2="0" y2="{-RADIUS * math.cos(ELEVATION):.2f}" '
        f'stroke="{colours["muted"]}" stroke-width="2"/>',
    ]
    for front, d in _runs(_path()):
        opacity = "1" if front else "0.4"
        parts.append(
            f'<path d="{d}" fill="none" stroke="{colours["amber"]}" stroke-width="3" '
            f'stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}"/>'
        )
    parts += [
        f'<line x1="0" y1="0" x2="{shaft_x:.2f}" y2="{shaft_y:.2f}" '
        f'stroke="{colours["blue"]}" stroke-width="6" stroke-linecap="round"/>',
        f'<polygon points="{head}" fill="{colours["blue"]}"/>',
        f'<circle r="4.5" fill="{colours["blue"]}"/>',
    ]
    return "\n".join(parts)


def _svg(view_box: str, title: str, desc: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}" role="img" '
        'aria-labelledby="title desc">\n'
        f'<title id="title">{title}</title>\n<desc id="desc">{desc}</desc>\n'
        f"{body}\n</svg>\n"
    )


def mark(colours: dict[str, str]) -> str:
    """The compact mark, used in the sidebar and as the favicon."""
    return _svg(
        "0 0 112 112",
        "blochsim",
        "A Bloch sphere with the magnetization vector at the end of its precessing excitation path",
        f'<g transform="translate(56 56)">\n{_mark(colours)}\n</g>',
    )


def logo(colours: dict[str, str]) -> str:
    """The mark followed by the wordmark."""
    word = (
        '<text x="128" y="96" font-family="Arial,Helvetica,sans-serif" font-size="96">'
        f'<tspan fill="{colours["blue"]}">bloch</tspan>'
        f'<tspan fill="{colours["amber"]}" font-weight="700">sim</tspan></text>'
    )
    return _svg(
        "0 0 520 128",
        "blochsim",
        "A Bloch sphere with a precessing magnetization vector, followed by the wordmark blochsim",
        f'<g transform="translate(60 64)">\n{_mark(colours)}\n</g>\n{word}',
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for theme, colours in PALETTE.items():
        suffix = "" if theme == "light" else "-dark"
        (OUT / f"blochsim-mark{suffix}.svg").write_text(mark(colours))
        (OUT / f"blochsim-logo{suffix}.svg").write_text(logo(colours))


if __name__ == "__main__":
    main()
