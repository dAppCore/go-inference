<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# reasoning-preservation

The gemma4 canonical-template reasoning-preservation rule (2026-07-09): a stateless client
replaying agentic history echoes back each assistant turn's chain of thought (OpenAI-wire
reasoning / reasoning_content); the serving layer re-frames it into the native thought span —
ALWAYS for a turn after the last user message (the live in-flight exchange keeps its own
thinking), and, under chat_template_kwargs.preserve_thinking, for a tool-calling assistant turn
ANYWHERE in history (an agentic tool loop keeps its reasoning across the replay). Earlier
plain-answer turns always replay clean — the model never sees stale chain-of-thought it didn't
just produce.

requestMessages (go/serving/provider/openai/handler.go), which does this re-framing, is
unexported — this example drives it the only way an external caller can: through the public
HTTP handler, httptest, and a fake model whose Chat method records exactly what it was called
with (the same fake-model pattern go/serving/compat's own tests use). One replayed agentic
history is POSTed twice — preserve_thinking false and true — and the two renders are compared.

## Run

```sh
go run ./pkg/reasoning-preservation
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
