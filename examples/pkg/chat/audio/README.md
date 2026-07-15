<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# chat/audio

Audio chat: attach a WAV clip (16-bit PCM mono 16 kHz — see the Audios
field doc) to a user turn. As with vision, accepting audio is a live
capability of the LOADED CHECKPOINT, not a family-wide guarantee, so probe
before sending.

## Run

```sh
go run ./pkg/chat/audio -model ~/models/gemma-4-e2b-it-4bit -audio clip.wav
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
