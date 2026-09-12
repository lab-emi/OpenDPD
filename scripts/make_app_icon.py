"""Render the Studio application icon (Dock / taskbar) from the favicon geometry.

    python scripts/make_app_icon.py     # writes opendpd/studio/icon.png and icon.ico

Same shapes as frontend/public/favicon.svg: a rounded #0B5FA5 square and a
white five-point polyline. Drawn 8x oversampled and downscaled for smooth edges.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "opendpd" / "studio"
SIZE, SCALE = 256, 8
BLUE, WHITE = (11, 95, 165, 255), (255, 255, 255, 255)
POINTS = [(6, 22), (11, 12), (16, 20), (21, 8), (26, 22)]     # favicon viewBox 0..32


def render() -> Image.Image:
    big = SIZE * SCALE
    unit = big / 32
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, big - 1, big - 1), radius=int(6 * unit), fill=BLUE)
    pts = [(x * unit, y * unit) for x, y in POINTS]
    width = int(3 * unit)
    draw.line(pts, fill=WHITE, width=width, joint="curve")
    for x, y in (pts[0], pts[-1]):          # round caps
        r = width / 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=WHITE)
    return img.resize((SIZE, SIZE), Image.LANCZOS)


def main() -> None:
    img = render()
    img.save(OUT / "icon.png")
    img.save(OUT / "icon.ico", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    print(f"wrote {OUT / 'icon.png'} and {OUT / 'icon.ico'}")


if __name__ == "__main__":
    main()
