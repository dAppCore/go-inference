# SPDX-Licence-Identifier: EUPL-1.2
#
# dump_reference.py regenerates bge_small_reference.json — the parity gold for
# the Go host BERT encoder. It runs BAAI/bge-small-en-v1.5 through the exact
# sentence-transformers pipeline (BertModel -> CLS token -> L2 normalise) using
# plain transformers, so the Go forward can be checked at cosine >= 0.999 per
# vector without shipping the 130MB checkpoint into the repo.
#
# Run (snapshot must already be in the HF cache):
#   /Users/snider/PyCharmMiscProject/.venv/bin/python \
#       model/bert/testdata/dump_reference.py
import json
import os
import sys

import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "BAAI/bge-small-en-v1.5"

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models can generate text embeddings.",
    "Lethean builds a sovereign local inference stack.",
    "How do I reset my password?",
    "Vector search retrieves semantically similar documents.",
]


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()

    records = []
    for text in SENTENCES:
        enc = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        # sentence-transformers CLS pooling: token 0 of last_hidden_state
        # (the pooler.dense/tanh head is NOT applied), then L2 normalise.
        cls = out.last_hidden_state[0, 0]
        vec = torch.nn.functional.normalize(cls, p=2, dim=0)
        records.append(
            {
                "text": text,
                "input_ids": enc["input_ids"][0].tolist(),
                "embedding": [float(x) for x in vec.tolist()],
            }
        )

    payload = {
        "model": MODEL,
        "pooling": "cls",
        "normalize": True,
        "hidden_size": model.config.hidden_size,
        "records": records,
    }
    out_path = os.path.join(os.path.dirname(__file__), "bge_small_reference.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("wrote", out_path, "records", len(records), "dim", model.config.hidden_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
