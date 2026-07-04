// SPDX-Licence-Identifier: EUPL-1.2

// Error-path benches. Validators + early-rejection paths run on every
// Load / Validate, so the cold dispatch cost matters. The target shape
// is a fast O(1) reject — these benches measure that and surface any
// path that allocates on a refusal (a common refactor regression).
//
// Coverage map (W7-F deepening pass):
//
//   - Snapshot.Save on nil snapshot (early NewError dispatch)
//   - Load on truncated header (Magic mismatch / version OOB)
//   - LoadWithOptions on truncated body (mid-stream parse failure)
//   - parseKVSnapshot on wrong magic — guards the State-bundle hash
//     mismatch surface.
//   - normalizeKVSnapshotEncoding on bad encoding string — fires per
//     Save/Hash on every checkpoint, so the rejection cost matters.
//   - ValidateStateBlockBundle on nil / version-OOB / wrong-kind /
//     zero-token / empty-blocks bundles.
//   - LoadFromStateBlocks on chunk-not-found store (the ChunkNotFound
//     dispatch path).
//
// Run: go test -bench='BenchmarkErrorpath' -benchmem -run='^$' ./go/kv

package kv

import (
	"context"
	"testing"

	state "dappco.re/go/inference/state"
)

// --- Snapshot save/load early-reject ---

func BenchmarkErrorpath_Save_NilSnapshot(b *testing.B) {
	var snap *Snapshot
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkErr = snap.Save("/dev/null")
	}
}

func BenchmarkErrorpath_MarshalBinary_NilSnapshot(b *testing.B) {
	var snap *Snapshot
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes, benchSinkErr = snap.MarshalBinary()
	}
}

func BenchmarkErrorpath_UnmarshalBinary_BadMagic(b *testing.B) {
	bad := []byte("WRONGMAGIC\x00\x00\x00\x00\x00\x00\x00\x00")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var out Snapshot
		benchSinkErr = out.UnmarshalBinary(bad)
	}
}

func BenchmarkErrorpath_UnmarshalBinary_TruncatedHeader(b *testing.B) {
	bad := []byte("MLXKV") // shorter than magic; magic compare itself fails
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var out Snapshot
		benchSinkErr = out.UnmarshalBinary(bad)
	}
}

func BenchmarkErrorpath_UnmarshalBinary_BadVersion(b *testing.B) {
	// Valid magic + out-of-range version byte run.
	bad := make([]byte, 12)
	copy(bad, kvSnapshotMagic)
	// version = 0xffffffff (LE) — outside [1, SnapshotVersion]
	bad[8], bad[9], bad[10], bad[11] = 0xff, 0xff, 0xff, 0xff
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var out Snapshot
		benchSinkErr = out.UnmarshalBinary(bad)
	}
}

func BenchmarkErrorpath_UnmarshalBinary_TruncatedPayload(b *testing.B) {
	// Take a valid encode and chop it off at the architecture header so
	// the parser exhausts mid-stream — the kvSnapshotReader.err path.
	snap := benchSnapshot(64)
	data, err := snap.bytes()
	if err != nil {
		b.Fatal(err)
	}
	truncated := data[:len(kvSnapshotMagic)+8] // magic + version + start of architecture-length
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var out Snapshot
		benchSinkErr = out.UnmarshalBinary(truncated)
	}
}

// --- Encoding-string rejection ---

func BenchmarkErrorpath_Save_UnsupportedEncoding(b *testing.B) {
	snap := benchSnapshot(64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes, benchSinkErr = snap.bytesWithOptions(SaveOptions{KVEncoding: Encoding("totally-not-a-real-encoding")})
	}
}

// --- StateBlockBundle validator rejections ---

func BenchmarkErrorpath_ValidateBundle_NilBundle(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkErr = ValidateStateBlockBundle(nil)
	}
}

func BenchmarkErrorpath_ValidateBundle_BadVersion(b *testing.B) {
	bundle := &StateBlockBundle{Version: 9999, Kind: StateBlockBundleKind, TokenCount: 1, Blocks: []StateBlockRef{{TokenCount: 1}}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkErr = ValidateStateBlockBundle(bundle)
	}
}

func BenchmarkErrorpath_ValidateBundle_BadKind(b *testing.B) {
	bundle := &StateBlockBundle{Version: 1, Kind: "totally-not-a-bundle-kind", TokenCount: 1, Blocks: []StateBlockRef{{TokenCount: 1}}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkErr = ValidateStateBlockBundle(bundle)
	}
}

func BenchmarkErrorpath_ValidateBundle_ZeroTokens(b *testing.B) {
	bundle := &StateBlockBundle{Version: 1, Kind: StateBlockBundleKind, TokenCount: 0, Blocks: []StateBlockRef{{TokenCount: 1}}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkErr = ValidateStateBlockBundle(bundle)
	}
}

func BenchmarkErrorpath_ValidateBundle_EmptyBlocks(b *testing.B) {
	bundle := &StateBlockBundle{Version: 1, Kind: StateBlockBundleKind, TokenCount: 64, Blocks: nil}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkErr = ValidateStateBlockBundle(bundle)
	}
}

// --- LoadFromStateBlocks against a store that doesn't have the chunks ---

func BenchmarkErrorpath_LoadStateBlocks_ChunkNotFound(b *testing.B) {
	// Build a valid bundle that references chunks that don't exist
	// in a fresh store. The error originates in
	// state.ResolveRefBytes → ChunkNotFoundError.
	emptyStore := state.NewInMemoryStore(nil)
	bundle := &StateBlockBundle{
		Version:      StateBlockVersion,
		Kind:         StateBlockBundleKind,
		Architecture: "qwen3",
		TokenCount:   64,
		TokenOffset:  64,
		BlockSize:    64,
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       64,
		HeadDim:      1,
		Blocks: []StateBlockRef{{
			Index:           0,
			TokenStart:      0,
			TokenCount:      64,
			PayloadEncoding: kvSnapshotStatePayloadRaw,
			State:           state.ChunkRef{ChunkID: 9999, Codec: state.CodecMemory},
		}},
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := LoadFromStateBlocks(ctx, emptyStore, bundle)
		if err == nil {
			b.Fatal("expected ChunkNotFound, got nil")
		}
		benchSinkSnapshot = out
		benchSinkErr = err
	}
}

// --- LoadFromState chunk-not-found dispatch ---

func BenchmarkErrorpath_LoadFromState_ChunkNotFound(b *testing.B) {
	emptyStore := state.NewInMemoryStore(nil)
	ref := state.ChunkRef{ChunkID: 9999, Codec: state.CodecMemory}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := LoadFromState(ctx, emptyStore, ref)
		if err == nil {
			b.Fatal("expected ChunkNotFound, got nil")
		}
		benchSinkSnapshot = out
		benchSinkErr = err
	}
}
