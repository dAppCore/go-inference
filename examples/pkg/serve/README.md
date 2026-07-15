<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# serve

Embed the OpenAI-compatible server in your own Go app: one call to
serving.RunServe hosts the same OpenAI/Anthropic/Ollama-compatible mux
`lem serve` does, but wired into a process you control (no cmd/lem
subprocess). RunServe blocks until ctx is cancelled, so Ctrl-C here shuts
the listener down cleanly.

curl http://127.0.0.1:36912/v1/chat/completions \
-H 'Content-Type: application/json' \
-d '{"model":"gemma-4-e2b-it-4bit","messages":[{"role":"user","content":"hi"}]}'

## Run

```sh
go run ./pkg/serve -model ~/models/gemma-4-e2b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
