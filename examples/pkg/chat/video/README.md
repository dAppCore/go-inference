<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# chat/video

Video chat: Message.Videos carries the sampled FRAMES of one video —
encoded PNG/JPEG images, in time order — rather than a video container
itself; the engine treats each frame as a timestamped vision block. This
example takes pre-extracted frame files (ffmpeg or similar does the
sampling) and attaches them in the order given.

## Run

```sh
go run ./pkg/chat/video -model ~/models/gemma-4-e2b-it-4bit -frames f1.jpg,f2.jpg,f3.jpg
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
