<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# chat examples

The conversational surface of `inference.TextModel.Chat` — each folder is one
feature, its README says what it shows, and its `main.go` is the whole example.

| example | shows |
|---------|-------|
| [basic](basic/) | one user turn, direct answer (thinking off) |
| [stream](stream/) | per-token streaming — the iterator IS the stream |
| [thinking](thinking/) | the Gemma 4 thought channel, split into thought + answer |
| [multiturn](multiturn/) | conversation history — the message slice IS the memory |
| [sampling](sampling/) | temperature/top-k/top-p/min-p/seed, seeded reproducibility |
| [stop](stop/) | stop tokens, suppression, min-tokens-before-stop |
| [cancel](cancel/) | cancelling generation: ctx cancel vs iterator break |
| [budget](budget/) | `WithThinkingBudget` — capped thought channel |
| [vision](vision/) | attach an image to a turn, with the capability probe |
| [audio](audio/) | attach WAV audio, with the `AudioModel` probe |
| [video](video/) | attach video frames in time order |
| [mtp](mtp/) | MTP speculative pair — target + assistant drafter (darwin/arm64) |
