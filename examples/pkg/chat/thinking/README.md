<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# chat/thinking

The Gemma 4 thought channel: with thinking ON (the model default) the raw
token stream carries the model's reasoning between channel markers, then
the visible answer:

<|channel>thought ...reasoning... <channel|> ...answer...

The serve layer splits this automatically (the reply's `thought` field);
a library caller sees the raw stream and splits it with the markers, as
below. Compare pkg/chat/basic, which disables the channel entirely.

## Run

```sh
go run ./pkg/chat/thinking -model ~/models/gemma-4-e2b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
