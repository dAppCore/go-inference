#!/usr/bin/env python3
# SPDX-Licence-Identifier: EUPL-1.2
#
# Generates fixture.png: the deterministic OCR end-to-end fixture committed alongside this file.
# Black text on a white background, PIL's built-in bitmap default font (no system-font dependency,
# so this is reproducible byte-for-byte across machines) at a fixed size.
#
# The canvas is exactly 1024x1024 -- DeepseekOCRForCausalLM.infer's "global view" ALWAYS letterboxes
# the input onto a 1024x1024 canvas via PIL's ImageOps.pad, which internally resizes with a bicubic
# filter whenever the source size differs from the target. Bicubic resampling is impractical to port
# bit-exactly into Go (kernel/rounding differences between PIL's C resampler and any Go image library
# are near-guaranteed), so this fixture sidesteps that non-goal entirely: generating it AT the exact
# target size makes ImageOps.pad's resize step a proven no-op (verified empirically: same-size
# ImageOps.pad/contain reproduces the input byte-for-byte, no resampling touches the pixels), so the
# Go port only needs a straight decode + normalise, and the E2E golden's image-tower input is pixel-
# identical between the Python reference and the Go port. ocr.go's doc comment names the general
# resize/letterbox path (non-1024x1024 input) as the v1 boundary this sidesteps.
#
# Regenerate with:
#   /Users/snider/PyCharmMiscProject/.venv/bin/python testdata/gen_fixture.py
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1024, 1024
TEXT = "Lethean OCR 2026"

img = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
draw = ImageDraw.Draw(img)
font = ImageFont.load_default(size=72)
bbox = draw.textbbox((0, 0), TEXT, font=font)
tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
x = (WIDTH - tw) // 2 - bbox[0]
y = (HEIGHT - th) // 2 - bbox[1]
draw.text((x, y), TEXT, font=font, fill=(0, 0, 0))
img.save("fixture.png")
print("wrote fixture.png", img.size)
