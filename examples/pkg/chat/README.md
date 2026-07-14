<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# chat examples

The conversational surface of `inference.TextModel.Chat` — each folder is one
feature, its README says what it shows, and its `main.go` is the whole example.

| example | shows |
|---------|-------|
| [basic](basic/) | one user turn, direct answer (thinking off) |
| [stream](stream/) | per-token streaming — the iterator IS the stream |
| [thinking](thinking/) | the Gemma 4 thought channel, split into thought + answer |
| [vision](vision/) | attach an image to a turn, with the capability probe |
