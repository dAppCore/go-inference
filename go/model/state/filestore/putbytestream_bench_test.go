// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the PutBytesStream backpressure surface.
// Per AX-11 — PutBytesStream is the streaming variant that lets the
// caller feed a payload of declared size through an io.Writer chain.
// The limitedPayloadWriter guards against over/under-write — every
// streamed Save runs through it. Sub-header, very-large, and chunked-
// write scenarios stress different parts of the path.
//
// Run:    go test -bench='BenchmarkFilestoreStream' -benchmem -run='^$' ./state/filestore

package filestore

import (
	"context"
	stdio "io"
	"testing"

	state "dappco.re/go/inference/model/state"
)

// Sinks defeat compiler DCE.
var (
	fsSinkRef state.ChunkRef
	fsSinkErr error
)

// --- Stream small payloads (sub-recordHeader-size) ---
// Single-byte writes are pathological for the limitedPayloadWriter —
// no batching benefit. Common for streamed metadata-only sentinels.

func BenchmarkFilestoreStream_OneByte(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/onebyte.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	opts := state.PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fsSinkRef, fsSinkErr = store.PutBytesStream(ctx, 1, opts, func(w stdio.Writer) error {
			_, err := w.Write([]byte{'a'})
			return err
		})
	}
}

func BenchmarkFilestoreStream_Sub16(b *testing.B) {
	// 16 bytes is smaller than recordHeaderLen (24). Confirms the
	// header write cost dominates a payload-size-tiny stream.
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/sub16.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	payload := []byte("0123456789abcdef")
	opts := state.PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.SetBytes(16)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fsSinkRef, fsSinkErr = store.PutBytesStream(ctx, len(payload), opts, func(w stdio.Writer) error {
			_, err := w.Write(payload)
			return err
		})
	}
}

// --- Stream large payloads (1MB, 4MB) ---
// Large state slices — a model-state checkpoint of a single KV layer
// can be MBs. The bench tracks the throughput floor.

func BenchmarkFilestoreStream_1MB(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/1mb.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	payload := make([]byte, 1024*1024)
	opts := state.PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.SetBytes(1024 * 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fsSinkRef, fsSinkErr = store.PutBytesStream(ctx, len(payload), opts, func(w stdio.Writer) error {
			_, err := w.Write(payload)
			return err
		})
	}
}

func BenchmarkFilestoreStream_4MB(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/4mb.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	payload := make([]byte, 4*1024*1024)
	opts := state.PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.SetBytes(4 * 1024 * 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fsSinkRef, fsSinkErr = store.PutBytesStream(ctx, len(payload), opts, func(w stdio.Writer) error {
			_, err := w.Write(payload)
			return err
		})
	}
}

// --- Chunked writes ---
// 4-chunk write of a 64KB payload — common shape when the caller
// streams from a buffered upstream reader. Each Write call costs
// one limitedPayloadWriter dispatch.

func BenchmarkFilestoreStream_Chunked_4x16KB(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/chunked.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	chunk := make([]byte, 16*1024)
	opts := state.PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.SetBytes(64 * 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fsSinkRef, fsSinkErr = store.PutBytesStream(ctx, 4*len(chunk), opts, func(w stdio.Writer) error {
			for range 4 {
				if _, err := w.Write(chunk); err != nil {
					return err
				}
			}
			return nil
		})
	}
}

func BenchmarkFilestoreStream_Chunked_16x4KB(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/chunked16.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	chunk := make([]byte, 4*1024)
	opts := state.PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.SetBytes(64 * 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fsSinkRef, fsSinkErr = store.PutBytesStream(ctx, 16*len(chunk), opts, func(w stdio.Writer) error {
			for range 16 {
				if _, err := w.Write(chunk); err != nil {
					return err
				}
			}
			return nil
		})
	}
}

// --- Stream-with-error-mid-write ---
// The writer returns an error part-way through. PutBytesStream must
// roll back the partial write + remove the orphan record. Fires on
// upstream EOF/cancellation paths.

func BenchmarkFilestoreStream_ErrorMidWrite(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/err.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	chunk := make([]byte, 1024)
	opts := state.PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, fsSinkErr = store.PutBytesStream(ctx, 4*len(chunk), opts, func(w stdio.Writer) error {
			// Write the first chunk, then bail. PutBytesStream must
			// reject because payloadWriter.remaining != 0 after the
			// callback returns nil-error. The "short-payload" path
			// exercises rollbackWriteLocked.
			_, _ = w.Write(chunk)
			return nil
		})
	}
}

// --- Stream-oversize-write ---
// The callback writes more bytes than declared. The limitedPayloadWriter
// rejects + rolls back.

func BenchmarkFilestoreStream_OversizeWrite(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/over.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	chunk := make([]byte, 1024)
	opts := state.PutOptions{Kind: "bench"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, fsSinkErr = store.PutBytesStream(ctx, 512, opts, func(w stdio.Writer) error {
			// Declared 512 but writes 1024 — limitedPayloadWriter rejects.
			_, err := w.Write(chunk)
			return err
		})
	}
}

// --- Stream-with-explicit-error ---
// The callback returns an error before writing. PutBytesStream must
// roll back the header that's already on disk.

func BenchmarkFilestoreStream_ExplicitError(b *testing.B) {
	dir := b.TempDir()
	store, err := Create(context.Background(), dir+"/explicit.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	opts := state.PutOptions{Kind: "bench"}
	sentinel := stdio.ErrShortBuffer
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, fsSinkErr = store.PutBytesStream(ctx, 64, opts, func(_ stdio.Writer) error {
			return sentinel
		})
	}
}
