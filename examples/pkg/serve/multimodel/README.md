<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# serve/multimodel

Embed the multi-model serve: two model snapshots held resident behind one
listener, selected per request by alias. serving.LoadModelsConfig reads the
same declarative --models-config JSON `lem serve -models-config` takes;
this example writes a minimal 2-model config from two flags instead of
requiring a hand-authored file (see serving.ModelsConfig for the full shape,
including memory ceiling and idle-TTL eviction).

curl http://127.0.0.1:36914/v1/chat/completions \
-H 'Content-Type: application/json' \
-d '{"model":"small","messages":[{"role":"user","content":"hi"}]}'
curl http://127.0.0.1:36914/v1/chat/completions \
-H 'Content-Type: application/json' \
-d '{"model":"big","messages":[{"role":"user","content":"hi"}]}'

## Run

```sh
go run ./pkg/serve/multimodel -model1 ~/models/gemma-4-e2b-it-4bit -model2 ~/models/qwen3-8b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
