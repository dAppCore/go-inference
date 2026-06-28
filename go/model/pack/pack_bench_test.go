// SPDX-Licence-Identifier: EUPL-1.2

package pack_test

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/pack"
)

// AX-11 baseline benchmarks for the model/pack public surface. This
// package owns the .model on-disk format every backend (go-mlx,
// go-rocm, go-cuda) shells through to ship + verify packed model
// artifacts. Per-Pack/Hash/Inspect cost matters because:
//
//   - Hash runs on every Pack() call (auto-populated into Manifest.Model.Hash)
//   - Fingerprint runs on every cross-machine identity check
//   - List + Inspect run on every "what's in this pack" CLI op + every
//     fleet-side compatibility sniff
//
// No bench coverage existed before this file. AX-11 § "Audit cadence":
// "New hot-path functions without accompanying benchmarks block merge."
// Landing the baseline IS the AX-11 contract.
//
// Run:
//   go test -bench=. -benchmem -benchtime=300ms ./model/pack/...

// sinks prevent the compiler from optimising bench bodies away.
var (
	packBenchSinkString  string
	packBenchSinkResult  core.Result
	packBenchSinkErr     error
	packBenchSinkEntries []pack.Entry
)

// --- Hash ---

// Hash on a typical fixture model dir — 4 metadata files + 1 fake
// safetensors file. Mirrors what go-mlx + go-rocm load when probing
// a local model.
func BenchmarkPack_Hash_Typical(b *testing.B) {
	tempRoot := (&core.Fs{}).NewUnrestricted().TempDir("pack-bench-hash-")
	defer core.RemoveAll(tempRoot)
	srcDir := core.JoinPath(tempRoot, "src")
	buildFixturePack(b, srcDir)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hash, r := pack.Hash(srcDir)
		packBenchSinkString = hash
		packBenchSinkResult = r
	}
}

// --- Fingerprint ---

// Fingerprint on a populated Manifest. Used for "is this the same
// logical model?" without reading the payload — fleet routing
// compatibility check.
func BenchmarkPack_Fingerprint_Typical(b *testing.B) {
	m := sampleManifest()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		packBenchSinkString = pack.Fingerprint(m)
	}
}

// --- List ---

// List on a packed model — manifest decode + entry enumeration. Used
// by lthn CLI's `pack list` verb and by inspector UIs.
func BenchmarkPack_List_Typical(b *testing.B) {
	tempRoot := (&core.Fs{}).NewUnrestricted().TempDir("pack-bench-list-")
	defer core.RemoveAll(tempRoot)
	srcDir := core.JoinPath(tempRoot, "src")
	dest := core.JoinPath(tempRoot, "out.model")
	buildFixturePack(b, srcDir)
	if r := pack.Pack(srcDir, dest, pack.PackOptions{Manifest: sampleManifest()}); !r.OK {
		b.Fatalf("Pack setup: %v", r.Value)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		entries, _, r := pack.List(dest)
		packBenchSinkEntries = entries
		packBenchSinkResult = r
	}
}

// --- Inspect ---

// Inspect on a packed model — manifest + structural inspection report.
// Slightly more work than List (also builds the inspection report).
func BenchmarkPack_Inspect_Typical(b *testing.B) {
	tempRoot := (&core.Fs{}).NewUnrestricted().TempDir("pack-bench-inspect-")
	defer core.RemoveAll(tempRoot)
	srcDir := core.JoinPath(tempRoot, "src")
	dest := core.JoinPath(tempRoot, "out.model")
	buildFixturePack(b, srcDir)
	if r := pack.Pack(srcDir, dest, pack.PackOptions{Manifest: sampleManifest()}); !r.OK {
		b.Fatalf("Pack setup: %v", r.Value)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, r := pack.Inspect(dest)
		packBenchSinkResult = r
	}
}

// --- Pack ---

// Pack on a typical fixture — buildTar + manifestToHeaderMap +
// trix.Encode + WriteFile. The manifest carries a pre-filled
// Model.Hash so this bench isolates the Pack-specific path rather
// than re-measuring Hash (which has its own bench above). Pack runs
// on every model bundling op for every backend.
func BenchmarkPack_Pack_Typical(b *testing.B) {
	tempRoot := (&core.Fs{}).NewUnrestricted().TempDir("pack-bench-pack-")
	defer core.RemoveAll(tempRoot)
	srcDir := core.JoinPath(tempRoot, "src")
	dest := core.JoinPath(tempRoot, "out.model")
	buildFixturePack(b, srcDir)
	m := sampleManifest()
	m.Model.Hash = "prefilled-to-isolate-pack-path-from-hash"
	m.Producer.Created = "2026-01-01T00:00:00Z"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		packBenchSinkResult = pack.Pack(srcDir, dest, pack.PackOptions{Manifest: m})
	}
}

// --- Unpack ---

// Unpack on a packed model — trix.Decode + extractTar (writes the
// payload files back to disk). Overwrite is set so the destination
// can be reused across iterations without per-iter dir churn.
func BenchmarkPack_Unpack_Typical(b *testing.B) {
	tempRoot := (&core.Fs{}).NewUnrestricted().TempDir("pack-bench-unpack-")
	defer core.RemoveAll(tempRoot)
	srcDir := core.JoinPath(tempRoot, "src")
	dest := core.JoinPath(tempRoot, "out.model")
	outDir := core.JoinPath(tempRoot, "out")
	buildFixturePack(b, srcDir)
	if r := pack.Pack(srcDir, dest, pack.PackOptions{Manifest: sampleManifest()}); !r.OK {
		b.Fatalf("Pack setup: %v", r.Value)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		packBenchSinkResult = pack.Unpack(dest, outDir, pack.UnpackOptions{Overwrite: true})
	}
}

// AX-11: alloc + behavioural budget gate for Hash on the typical
// fixture. Hash runs on every Pack() call — a regression here
// propagates to model save time + drives up Pack latency for every
// backend that bundles models.
//
// Baseline measurement (Apple M3 Ultra, -benchmem): set after first
// run. The const below ratchets down as wins land.
func TestAllocBudget_Pack_Hash_Typical(t *testing.T) {
	tempRoot := (&core.Fs{}).NewUnrestricted().TempDir("pack-budget-hash-")
	defer core.RemoveAll(tempRoot)
	srcDir := core.JoinPath(tempRoot, "src")
	buildFixturePack(t, srcDir)

	// Behavioural lock — hash is deterministic for the same source
	// tree. Run twice + assert equal. Any future refactor that
	// quietly changes the hash function or input order fails loud.
	h1, r1 := pack.Hash(srcDir)
	if !r1.OK {
		t.Fatalf("Hash run 1: %v", r1.Value)
	}
	h2, r2 := pack.Hash(srcDir)
	if !r2.OK {
		t.Fatalf("Hash run 2: %v", r2.Value)
	}
	if h1 != h2 {
		t.Fatalf("Hash non-deterministic: %s != %s", h1, h2)
	}
	if len(h1) != 64 {
		t.Fatalf("expected 64-char sha256 hex, got %d chars", len(h1))
	}

	avg := testing.AllocsPerRun(5, func() {
		_, _ = pack.Hash(srcDir)
	})
	// Ceiling: 120 — current 112 (post sharedFs cache) + ~7% headroom.
	// Was 116→130 pre-sharedFs. Ratchet DOWN as optimisations land.
	// Remaining floor is OS file I/O (Stat, ReadFile, WalkSeq) — those
	// are below this layer and need bigger architectural moves to cut
	// further (mmap, single-syscall directory walk, etc).
	const budget = 120.0
	if avg > budget {
		t.Fatalf("pack.Hash alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"Hash runs on every Pack() — every backend pays this per model bundling op.\n"+
			"Profile: go test -bench=BenchmarkPack_Hash_Typical -benchmem -memprofile=/tmp/h.mem",
			avg, budget)
	}
}
