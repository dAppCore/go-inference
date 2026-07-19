#!/usr/bin/env python3
"""Generates the deterministic GLM-OCR fixture: 112x112 (the smart_resize
min-pixels self-mapping size — 112 = 4*28, and 2*112*112 > min_pixels=12544,
so GLM-OCR's image processor resizes it to itself, letting the Go port skip
bicubic resampling and still match the real pipeline exactly), black text on
a white background, deterministic (fixed size, fixed font, fixed text).
"""
from PIL import Image, ImageDraw, ImageFont

W = H = 112
img = Image.new("RGB", (W, H), (255, 255, 255))
draw = ImageDraw.Draw(img)
font = ImageFont.load_default()
text = "LEM OCR"
bbox = draw.textbbox((0, 0), text, font=font)
tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
draw.text(((W - tw) / 2 - bbox[0], (H - th) / 2 - bbox[1]), text, fill=(0, 0, 0), font=font)

out = "/private/tmp/claude-501/-Users-snider-Lethean-agent-cladius/4f1049ad-ecbc-425a-aacc-1ca2b340b3c9/scratchpad/glmocr/fixture.png"
img.save(out)
print("wrote", out, img.size, img.mode)
