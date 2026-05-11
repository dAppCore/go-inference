<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# state/memory.go — InMemoryStore

**Package**: `dappco.re/go/inference/state`
**File**: `go/state/memory.go`

## What this is

The in-process reference implementation of every read and write
interface in `state/store.go`. Maps `chunk_id → text|bytes` plus an
optional `uri → chunk_id` index. Zero file I/O, zero network, zero
codec — useful for tests, fixtures, and the "spike before wiring
memvid" path.

## Capabilities implemented

`*InMemoryStore` satisfies:

- `Store` (`Get`)
- `Resolver` (`Resolve`)
- `BinaryResolver` (`ResolveBytes`)
- `URIResolver` (`ResolveURI`)
- `Writer` (`Put`)
- `BinaryWriter` (`PutBytes`)

Not implemented:

- `RefBinaryResolver` (falls back to `ResolveBytes(chunk_id)`)
- `BinaryStreamWriter` (in-memory has no streaming win)

## Constructors

```go
state.NewInMemoryStore(map[int]string{1: "hello"})
state.NewInMemoryStoreWithManifest(chunks, refs)  // pre-seed ChunkRef metadata
```

The "WithManifest" form is for round-tripping fixtures — you write some
chunks via `Put`, capture the returned refs, then in a later test
recreate the same store with both the text *and* the refs so chunk-id
+ codec match.

## Codec stamp

Every ref written by this store carries `Codec: state.CodecMemory` and
`HasFrameOffset: true` with `FrameOffset == ChunkID`. The frame-offset
mirror makes test fixtures behave the same as memvid bundles for code
that branches on frame addressing — the test path doesn't need a
separate "I'm in fixture mode" flag.

## When NOT to use

This store is not safe across goroutines without external locking. A
production session uses memvid (file-backed, immutable) or filestore
(append-only on disk) for durability. Use `InMemoryStore` for:

- Unit tests against `Resolve` / `ResolveURI` / `Put`
- Fixture seeding in example tests
- Dev workflow where the wake/sleep loop runs in-process

## Consumed by

- `state/state_test.go` — round-trip + URI-resolution tests
- `go-mlx/agent_memory_test.go` — runtime smoke tests against a known
  in-memory store before reaching for memvid
- `go-ai/ai/book_state_demo_test.go` — bookstate fixtures point at
  in-memory chunks via `entry-uri memory://...`
