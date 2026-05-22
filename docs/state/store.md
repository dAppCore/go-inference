<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# state/store.go — chunk-addressable storage interfaces

**Package**: `dappco.re/go/inference/state`
**File**: `go/state/store.go`

## What this is

The portable contract for **chunk-addressable storage** that backs the
wake/sleep lifecycle. A bundle written by `Session.SleepState` becomes a
sequence of chunks behind one of these interfaces; a wake reads them
back via `Resolve` / `ResolveBytes` / `ResolveURI`.

Five storage capabilities expressed as separate, narrow interfaces. A
backend implements only what it can support — `Store.Get` for text,
`BinaryResolver` for bytes, `URIResolver` for State URI lookup,
`Writer` / `BinaryWriter` / `BinaryStreamWriter` for the encode side.

## Codecs

```go
CodecMemory  = "memory/plaintext"   // in-process test/dev store
CodecStateVideo = "state/qr-video"    // QR-encoded MP4 cold storage
```

The codec field on a `ChunkRef` tells the wake side which decoder to
spin up. State video is the portable `.mp4` codec; in-memory is the
test harness; filestore is the raw local file log.

## Capability matrix

| Interface | Read mode | Notes |
|-----------|-----------|-------|
| `Store` | text only | minimum viable backend |
| `Resolver` | text + ref metadata | upgrades a Store with offset info |
| `BinaryResolver` | bytes | for non-text bundles (KV blocks, attention snapshots) |
| `RefBinaryResolver` | bytes via `ChunkRef` | lets the store choose chunk id OR frame offset OR segment hint |
| `URIResolver` | bytes via `uri` | for stores that index by external URI rather than int id |

| Interface | Write mode | Notes |
|-----------|-----------|-------|
| `Writer` | text | smallest write surface |
| `BinaryWriter` | bytes in one buffer | the common path |
| `BinaryStreamWriter` | bytes via callback | for large bundles where buffering the whole payload would OOM the encoder |

The package-level free functions (`Resolve`, `ResolveBytes`,
`ResolveRefBytes`, `ResolveURI`) take a generic `Store` and probe up to
the richer interface via type assertion — so callers always get bytes if
they ask for bytes, even when only text is implemented.

## DTOs

`Chunk` — what comes back from a read:

```go
type Chunk struct {
    Ref  ChunkRef
    Text string   // empty for binary-only chunks
    Data []byte   // empty for text-only chunks (filled when caller asks ResolveBytes)
}
```

`ChunkRef` — the durable handle:

```go
type ChunkRef struct {
    ChunkID        int     // monotonic id within a bundle
    FrameOffset    uint64  // for State video: which video frame
    HasFrameOffset bool    // distinguishes "frame 0" from "unset"
    Codec          string  // state/qr-video, memory/plaintext, …
    Segment        string  // optional sub-segment id within the chunk
}
```

`PutOptions` — write-side metadata that the encoder retains alongside
bytes:

```go
type PutOptions struct {
    URI    string
    Title  string
    Kind   string                // "kv-block", "attention-snapshot", "prompt", …
    Track  string                // sub-stream within a bundle
    Tags   map[string]string
    Labels []string
}
```

## Errors

Two typed errors, both unwrapping to `ErrChunkNotFound`:

- `ChunkNotFoundError{ID: int}` — chunk-id miss
- `URIChunkNotFoundError{URI: string}` — URI-keyed miss

Callers use `errors.Is(err, state.ErrChunkNotFound)` to handle both
shapes uniformly.

## MergeRef

`MergeRef(base, overlay ChunkRef)` is the merge primitive used when a
bundle's index is updated incrementally — overlay non-zero fields, keep
base for the rest. Lets sleep-with-parent operations carry forward the
parent's chunk identity while updating frame offsets.

## Why not one big Store interface

Backends differ in what they can do. A full State video store implements every interface.
A test fixture might implement only `Store.Get`. The current `inference`
package code does type-assertion probing rather than forcing every
backend to stub out methods it can't actually perform — which means a
small backend can be 50 lines, not 500.

## Implemented by

- `state/memory.go` — `InMemoryStore`. Test fixture + dev workflow.
- `state/filestore/store.go` — raw file log (planned canonical for
  CoreAgent on-disk bundles).
- `go-mlx/pkg/memvid/filestore` — deprecated compatibility path.

## Consumed by

- `state/agent_memory.go` — Wake/Sleep/Fork hold a `Store any` and dial
  through these interfaces
- `go-mlx/pkg/memvid` — deprecated compatibility import path for older
  encoder/decoder callers
