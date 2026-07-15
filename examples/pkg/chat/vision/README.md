<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# chat/vision

Vision chat: attach an image (PNG/JPEG bytes) to a user turn. Whether the
LOADED CHECKPOINT accepts images is a live capability probe — the family
supporting vision does not mean this snapshot shipped the tower — so probe
before sending, exactly as the serve layer does.

## Run

```sh
go run ./pkg/chat/vision -model ~/models/gemma-4-e2b-it-4bit -image cat.png
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
