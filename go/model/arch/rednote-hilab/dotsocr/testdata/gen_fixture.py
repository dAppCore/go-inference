#!/usr/bin/env python3
# SPDX-Licence-Identifier: EUPL-1.2
#
# gen_fixture.py deterministically generates fixture.png: a white 280x84 RGB image with the
# fixed string FIXTURE_TEXT rendered in black using Pillow's built-in bitmap font (ImageFont.
# load_default — no external font file, so this reproduces identically on any machine with
# Pillow installed; no antialiasing/hinting variance from a system TTF).
#
# 280x84 is deliberately an exact multiple of patch_size*spatial_merge_size (14*2=28) on both
# axes (280=10*28, 84=3*28) — DOTS-OCR's Qwen2VLImageProcessor-style smart_resize leaves an
# already-aligned image untouched (see ../image.go's doc comment), so this fixture exercises
# the REAL preprocessing pipeline without going through the (approximate, non-bit-exact)
# resampling path.
#
# Regenerate with: python3 gen_fixture.py
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 280, 84
FIXTURE_TEXT = "Lethean OCR 2026"


def main() -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=24)
    bbox = draw.textbbox((0, 0), FIXTURE_TEXT, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (WIDTH - text_w) // 2 - bbox[0]
    y = (HEIGHT - text_h) // 2 - bbox[1]
    draw.text((x, y), FIXTURE_TEXT, fill=(0, 0, 0), font=font)
    img.save("fixture.png")
    print(f"wrote fixture.png ({WIDTH}x{HEIGHT}, text={FIXTURE_TEXT!r})")


if __name__ == "__main__":
    main()
