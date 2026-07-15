<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# tools

The full function-calling round trip on a Gemma 4 checkpoint: declare a
tool, let the model decide to call it, run the call locally, and feed the
result back for a final answer — the "lem can call your Go functions"
story.

Gemma 4's native tool syntax is a special-token grammar the decoder
preserves verbatim (decode/parser/gemma_tools.go); this example renders a
declaration, parses the model's own call span out of its reply, and wraps
the tool's answer in the matching <|tool_response> span before asking for
the final turn.

## Run

```sh
go run ./pkg/tools -model ~/models/gemma-4-e2b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
