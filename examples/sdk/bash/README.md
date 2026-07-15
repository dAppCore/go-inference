# Use Gemma 4 from Bash — local OpenAI-compatible API (lem / go-inference)

No SDK runtime at all: the `bash` generator emits a self-contained curl
client (`lem-cli.sh`) with per-operation help and tab completion, and this
demo drives it past hello world — models, a two-turn conversation that
proves memory, the thinking toggle, and usage.

```bash
task sdk                    # once: generate build/sdk/bash
./examples/sdk/bash/chat.sh # against a running lem serve (LEM_BASE_URL overrides)
```

Direct use of the generated client:

```bash
bash build/sdk/bash/lem-cli.sh --host http://localhost:36911 getV1Models
echo '{"model":"gemma4","messages":[{"role":"user","content":"hello"}]}' \
  | bash build/sdk/bash/lem-cli.sh --host http://localhost:36911 \
      --content-type application/json postV1ChatCompletions -
bash build/sdk/bash/lem-cli.sh postV1ChatCompletions --help   # rendered from the spec
```

## Friction

- The generator's stated limitations (no oneOf/anyOf, no typed models) don't
  bite here — the client passes JSON through verbatim and the spec's raw
  response schemas document what comes back.
- A request body is passed as `-` (stdin) — not obvious from `--help`, which
  lists the parameter only as `body [application/json] (required)`.
- The demo leans on `jq` for parsing; that's a demo choice, not a client
  requirement.
- This lane found a real serving defect on its first run: a `--scheduler
  serial` serve ignored `chat_template_kwargs {"enable_thinking": false}`
  (fixed in the same session — the serial/batch dispatchers now re-arm the
  override like the interleave lane always did).
