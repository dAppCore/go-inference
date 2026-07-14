# cli — the `lem` binary

The command-line front end for go-inference: one Go module
(`dappco.re/go/inference/cli`) whose root package builds the binary. The
binary's name and version are build-time choices, not code — the Taskfile's
`BIN_NAME`/`VERSION` vars and the Makefile's `CLI_NAME`/`CLI_VERSION` thread
them (`-X main.version=…`), and usage lines print whatever argv[0] says, so a
renamed copy introduces itself correctly.

```bash
task build          # bin/lem for this machine (Apple: metal engine embedded)
./bin/lem serve --model ~/models/gemma-4-e2b-it-4bit   # OpenAI/Anthropic/Ollama API on :36911
./bin/lem tui       # chat with a model in the terminal
./bin/lem -h        # the full verb list
```

Each verb is thin flag-parsing over a go-inference library — the business
logic lives in the library packages (`serving`, `train`, `pack`, …), never
here.

| Path | What it is |
|------|-----------|
| `main.go`, `serve.go`, `generate.go`, … | one file per verb, flags → library call |
| `tui/` | the terminal UI (`lem tui`) — Bubble Tea tabs: Chat · Models · Service · Settings · Tools · Modes |
| `lthn-model-pack/` | standalone `.model` container helper binary |
| `*.metallib.gz` | embedded Metal kernel libraries (`-tags embed_metallib`) |
