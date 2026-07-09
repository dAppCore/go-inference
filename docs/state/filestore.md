<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# state/filestore — append-only file-backed state store

**Package**: `dappco.re/go/inference/model/state/filestore`
**File**: `go/model/state/filestore/store.go`

## What this is

A durable, single-file, append-only implementation of the `state.Store`
interfaces. Designed as the on-disk canonical for CoreAgent bundles
when State video packaging isn't required (most local-only
sessions). Each chunk is a self-describing record; the file as a whole
forms a write-ahead-log style history.

## File format

```
+-----------------------------------+
| FILE MAGIC                        |  "go-inference-state-file-log-v1\n" (31 bytes)
|                                   |  legacy: "go-mlx-memvid-file-log-v1\n" (26 bytes)
+-----------------------------------+
| Record 1                          |
|  - magic "MVF1"  (4)              |
|  - chunk_id     (8)               |
|  - payload size (8)               |
|  - meta size    (4)               |
|  - payload bytes ...              |
|  - meta JSON bytes ...            |
+-----------------------------------+
| Record 2 ...                      |
+-----------------------------------+
```

`recordHeaderLen = 24` (4 + 8 + 8 + 4). The full record header tells
the reader exactly how many bytes to seek over for the payload and how
many for the JSON-encoded metadata.

## Codec stamp

```go
const CodecFile = "state/file-log"
```

Bundles emitted by this store identify with `Codec: CodecFile` so a
wake on a State-video-only build can detect-and-route or refuse-and-warn
based on whether the file-log decoder is compiled in.

## Backward compatibility

The legacy magic `go-mlx-memvid-file-log-v1\n` is still recognised on
open — older bundles written when this code lived in `go-mlx`
round-trip without rewrite. New writes always use the
`go-inference-state-file-log-v1\n` magic.

## API

```go
filestore.Create(ctx, path) (*Store, error)     // new file
filestore.Open(ctx, path)   (*Store, error)     // read existing, rebuild index in RAM
```

(`OpenWithSegmentAlias` and `OpenRegionWithSegmentAlias` handle relocated
or region-embedded State files — see the source.)

Once open, `*Store` satisfies `state.Store` + `state.Resolver` +
`state.URIResolver` + `state.Writer` + `state.BinaryWriter`. Index is
held in-memory; very large bundles benefit from a future on-disk
index — currently every URI/chunk-id lookup is O(1) hash but the index
itself is O(N) memory.

## Concurrency

One `sync.Mutex` per `Store`. Writes append at `writeAt`, reads scan
the index then `ReadAt` from the file. Multiple goroutines can read
concurrently with one writer holding the mutex during the
append-and-fsync.

## Failure modes

Append-only means a crash mid-write leaves a torn record at EOF. Open
detects truncated records (header reads past EOF or payload+meta short
of declared size) and rolls `writeAt` back to the last good record —
the partial bytes are overwritten on the next Put.

## When to use

- Local development without a State video encoder configured
- Single-machine CoreAgent that doesn't need portable .mp4 packs
- Test fixtures that need on-disk durability between processes

## When NOT to use

- Cross-machine bundle sharing → State video (`.mp4`)
- Object-storage backed bundles → S3 + custom resolver
- Read-mostly cold storage → State video (compression + scan-friendly)

## Consumed by

- `cmd/lem` — when configured with a local bundles directory
- `model/state/session/` — preferred Store for the Wake/Sleep loop
  when State video output isn't requested
- Test harnesses that need cross-test persistence (filestore lives,
  in-memory dies on process exit)
