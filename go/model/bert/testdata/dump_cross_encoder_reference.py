#!/usr/bin/env python3
# SPDX-Licence-Identifier: EUPL-1.2
"""Regenerate the ms-marco cross-encoder score receipt.

Tooling: python -m pip install torch transformers; then run this file. The model
is https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2 and its default
activation is Identity, so the printed values are raw classifier logits.
"""
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY = "How do I reset my password?"
DOCUMENTS = [
    "The quick brown fox jumps over the lazy dog.",
    "To change your password, open account settings and choose reset password.",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
with torch.no_grad():
    inputs = tokenizer([QUERY] * len(DOCUMENTS), DOCUMENTS, padding=True, return_tensors="pt")
    scores = model(**inputs).logits[:, 0]
print(json.dumps({
    "model": MODEL,
    "source": f"https://huggingface.co/{MODEL}",
    "query": QUERY,
    "documents": DOCUMENTS,
    "scores": [float(score) for score in scores],
}, indent=2))
