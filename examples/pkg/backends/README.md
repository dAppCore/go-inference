<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# backends

The backend registry: every engine that blank-imports itself registers with
inference.Register at init() time, and inference.List/All/Get/Default read
that registry back. This example blank-imports both engines this repo
ships — the platform engine (metal on darwin/arm64, via
examples/internal/engine) and engine/hip (registers "rocm"; off
linux/amd64 it registers the honest stub with Available() false rather
than not registering at all) — so the printed list is platform-truthful:
what you see is what would actually load on THIS machine, not a hardcoded
menu.

## Run

```sh
go run ./pkg/backends                              # zero-hardware: print the registry, exit
go run ./pkg/backends -model ~/models/gemma-4-e2b-it-4bit  # also load, pinned to the default backend
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
