#!/usr/bin/env python3
"""Generate assets/sinribe.ico — the icon on the desktop shortcut and in the title bar.

Standard library only (zlib + struct), so the icon can be regenerated on any machine without a
build environment or an image library. It draws a rounded square in Sinribe's accent blue with a
white microphone on it, supersampled 4x for smooth edges, and writes every size Windows asks for
into a single .ico.

    python tools/make_icon.py            rewrite assets/sinribe.ico
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "assets" / "sinribe.ico"
SIZES = (16, 24, 32, 48, 64, 128, 256)
SS = 4  # supersampling factor per axis

BG_TOP = (0x6E, 0xA1, 0xFF)
BG_BOTTOM = (0x3B, 0x6F, 0xE0)
GLYPH = (0xFF, 0xFF, 0xFF)


def _rounded(x: float, y: float, n: float, radius: float) -> bool:
    """Inside a rounded square spanning 0..n?"""
    cx = min(max(x, radius), n - radius)
    cy = min(max(y, radius), n - radius)
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2


def _mic(x: float, y: float, n: float) -> bool:
    """A microphone: capsule, the arc cradling it, and the stand.

    Coordinates are fractions of the icon so the shape holds at every size.
    """
    fx, fy = x / n, y / n
    # capsule body
    cap_w, cap_top, cap_bot = 0.105, 0.235, 0.545
    if abs(fx - 0.5) <= cap_w and cap_top <= fy <= cap_bot:
        return True
    for cy_ in (cap_top, cap_bot):
        if (fx - 0.5) ** 2 + (fy - cy_) ** 2 <= cap_w ** 2:
            return True
    # cradle arc: an annulus, lower half only
    r = ((fx - 0.5) ** 2 + (fy - 0.5) ** 2) ** 0.5
    if 0.205 <= r <= 0.265 and fy >= 0.495:
        return True
    # stand
    if abs(fx - 0.5) <= 0.032 and 0.70 <= fy <= 0.80:
        return True
    if abs(fx - 0.5) <= 0.135 and 0.775 <= fy <= 0.812:
        return True
    return False


def render(n: int) -> bytes:
    """RGBA bytes for one n x n icon."""
    rows = []
    radius = n * 0.22
    for py in range(n):
        row = bytearray()
        for px in range(n):
            r = g = b = a = 0.0
            for sy in range(SS):
                for sx in range(SS):
                    x = px + (sx + 0.5) / SS
                    y = py + (sy + 0.5) / SS
                    if not _rounded(x, y, n, radius):
                        continue
                    t = y / n
                    br = BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * t
                    bg = BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * t
                    bb = BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * t
                    if _mic(x, y, n):
                        br, bg, bb = GLYPH
                    r += br
                    g += bg
                    b += bb
                    a += 255.0
            k = SS * SS
            # Premultiplied averaging would darken the anti-aliased rim against a light
            # background; divide the colour by the covered samples instead of by all of them.
            cov = a / 255.0
            if cov > 0:
                row += bytes((int(r / cov), int(g / cov), int(b / cov), int(a / k)))
            else:
                row += b"\x00\x00\x00\x00"
        rows.append(bytes(row))
    return b"".join(rows)


def png(n: int, rgba: bytes) -> bytes:
    """Minimal PNG encoder: one IHDR, one IDAT, one IEND."""
    raw = b"".join(b"\x00" + rgba[y * n * 4:(y + 1) * n * 4] for y in range(n))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def build_ico(sizes=SIZES) -> bytes:
    """ICO container holding a PNG per size (supported by Windows Vista and later)."""
    images = [png(n, render(n)) for n in sizes]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = len(header) + 16 * len(images)
    entries, blobs = b"", b""
    for n, data in zip(sizes, images, strict=True):
        entries += struct.pack("<BBBBHHII", n if n < 256 else 0, n if n < 256 else 0,
                               0, 0, 1, 32, len(data), offset)
        offset += len(data)
        blobs += data
    return header + entries + blobs


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = build_ico()
    OUT.write_bytes(data)
    print(f"wrote {OUT} ({len(data) / 1024:.1f} KB, sizes {', '.join(map(str, SIZES))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
