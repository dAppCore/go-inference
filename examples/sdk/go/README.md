# sdk/go — gemma4 via the generated Go client

```bash
task sdk                       # once: generate build/sdk/go
cd examples/sdk/go
GOWORK=off go run .            # against a running lem serve (LEM_BASE_URL overrides)
```

The client is `dappco.re/go/inference/lemsdk`, generated from the OpenAPI
spec — typed request builders, typed `choices[].message.{content,thought}`
response, no hand-written HTTP. `GOWORK=off` keeps the repo workspace out of
the way — this demo module deliberately depends on generated (gitignored)
output, so it can't live in go.work.
