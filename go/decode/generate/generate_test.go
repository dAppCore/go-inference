// SPDX-Licence-Identifier: EUPL-1.2

package generate

import (
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/serving"
)

// capModel is a fake inference.TextModel that reports a chosen capability set,
// so the -kv-cache note logic can be exercised without loading an engine.
type capModel struct {
	inference.TextModel
	report inference.CapabilityReport
}

func (m capModel) Capabilities() inference.CapabilityReport { return m.report }

// TestNoteCacheKnobs_UnhonouredMode_Good pins the honest note: a -kv-cache mode
// the engine does not declare is reported ignored, naming what IS supported.
func TestNoteCacheKnobs_UnhonouredMode_Good(t *testing.T) {
	log := core.NewBuffer()
	tm := capModel{report: inference.CapabilityReport{CacheModes: []string{"native"}}}
	noteCacheKnobs(Config{KVCacheMode: "q8", Log: log}, tm)
	out := log.String()
	if !core.Contains(out, "-kv-cache \"q8\"") || !core.Contains(out, "native") {
		t.Fatalf("note = %q, want it to name q8 as ignored and native as supported", out)
	}
}

// TestNoteCacheKnobs_HonouredMode_Bad pins the silence: a mode the engine DOES
// declare produces no note (it would be honoured).
func TestNoteCacheKnobs_HonouredMode_Bad(t *testing.T) {
	log := core.NewBuffer()
	tm := capModel{report: inference.CapabilityReport{CacheModes: []string{"native"}}}
	noteCacheKnobs(Config{KVCacheMode: "native", Log: log}, tm)
	if out := log.String(); out != "" {
		t.Fatalf("honoured mode noted %q, want silence", out)
	}
}

// TestNoteCacheKnobs_IgnoresStorage_Ugly pins that noteCacheKnobs is now
// -kv-cache-only: -kv-storage is the snapshot-encoding knob resolved elsewhere,
// so noteCacheKnobs stays silent about it even when set.
func TestNoteCacheKnobs_IgnoresStorage_Ugly(t *testing.T) {
	log := core.NewBuffer()
	tm := capModel{report: inference.CapabilityReport{CacheModes: []string{"native"}}}
	noteCacheKnobs(Config{KVStorage: "q8", Log: log}, tm)
	if out := log.String(); out != "" {
		t.Fatalf("noteCacheKnobs spoke about -kv-storage: %q, want silence", out)
	}
}

// TestKVStorageEncoding pins the flag → kv.Encoding resolution + classification:
// native is recognised AND live-state-ready; q8/float32 are recognised but NOT
// live-state-ready (they need the snapshot-codec follow-up, so they fall back to
// native); an unknown value is neither and also falls back to native.
func TestKVStorageEncoding(t *testing.T) {
	cases := []struct {
		raw        string
		enc        kv.Encoding
		recognised bool
		liveReady  bool
	}{
		{"", kv.EncodingNative, true, true},
		{"native", kv.EncodingNative, true, true},
		{"Q8", kv.EncodingQ8, true, false},
		{"float32", kv.KVSnapshotEncodingFloat32, true, false},
		{"fp16", kv.EncodingNative, false, false}, // go-mlx-era vocabulary, not a real encoding here
	}
	for _, c := range cases {
		enc, recognised, liveReady := kvStorageEncoding(c.raw)
		if enc != c.enc || recognised != c.recognised || liveReady != c.liveReady {
			t.Fatalf("kvStorageEncoding(%q) = (%q, %v, %v), want (%q, %v, %v)",
				c.raw, enc, recognised, liveReady, c.enc, c.recognised, c.liveReady)
		}
	}
}

// TestNoteKVStorageInert_Good pins the stateless-path note: a set -kv-storage
// value is reported inert (no KV persisted) so a bench user is not misled.
func TestNoteKVStorageInert_Good(t *testing.T) {
	log := core.NewBuffer()
	noteKVStorageInert(Config{KVStorage: "q8", Log: log})
	if out := log.String(); !core.Contains(out, "no effect here") {
		t.Fatalf("inert note = %q, want it to state -kv-storage has no effect statelessly", out)
	}
}

// TestNoteKVStorageInert_Bad pins the silence when the flag is unset.
func TestNoteKVStorageInert_Bad(t *testing.T) {
	log := core.NewBuffer()
	noteKVStorageInert(Config{Log: log})
	if out := log.String(); out != "" {
		t.Fatalf("unset -kv-storage noted %q, want silence", out)
	}
}

// TestNoteKVStorageInert_Ugly pins the unknown-value flag on the stateless path:
// an unrecognised encoding is named alongside the inert note.
func TestNoteKVStorageInert_Ugly(t *testing.T) {
	log := core.NewBuffer()
	noteKVStorageInert(Config{KVStorage: "fp16", Log: log})
	out := log.String()
	if !core.Contains(out, "not a known KV snapshot encoding") || !core.Contains(out, "no effect here") {
		t.Fatalf("unknown+inert note = %q, want both the unknown flag and the inert note", out)
	}
}

// TestCacheModeHonoured pins the case-insensitive membership the note keys on.
func TestCacheModeHonoured(t *testing.T) {
	if !cacheModeHonoured([]string{"Native"}, "native") {
		t.Fatal("cacheModeHonoured should match case-insensitively")
	}
	if cacheModeHonoured([]string{"native"}, "q8") {
		t.Fatal("cacheModeHonoured matched an absent mode")
	}
	if cacheModeHonoured(nil, "native") {
		t.Fatal("empty mode list honours nothing")
	}
}

// TestCacheModesSuffix pins the supported-modes hint rendering.
func TestCacheModesSuffix(t *testing.T) {
	if got := cacheModesSuffix(nil); got != "" {
		t.Fatalf("empty suffix = %q, want empty", got)
	}
	if got := cacheModesSuffix([]string{"native", "paged"}); !core.Contains(got, "native, paged") {
		t.Fatalf("suffix = %q, want it to list the modes", got)
	}
}

// TestPrintDecodePhaseBudget pins the rendered table: the header tok/s, the
// GPU-busy / host-serial split, and each named phase line.
func TestPrintDecodePhaseBudget(t *testing.T) {
	out := core.NewBuffer()
	budget := &inference.DecodePhaseBudget{
		Tokens:        10,
		TotalPerToken: 5 * time.Millisecond,
		GPUPerToken:   3 * time.Millisecond,
		Phases: []inference.DecodePhaseShare{
			{Name: "chained GPU span", PerToken: 3 * time.Millisecond, GPU: true},
		},
	}
	printDecodePhaseBudget(out, budget)
	s := out.String()
	for _, want := range []string{"10 tokens", "200.0 tok/s", "GPU busy", "host serial", "chained GPU span"} {
		if !core.Contains(s, want) {
			t.Fatalf("budget table missing %q:\n%s", want, s)
		}
	}
}

// TestTokPerSec pins the per-token-ms → tok/s conversion, including the guard.
func TestTokPerSec(t *testing.T) {
	if got := tokPerSec(5); got != 200 {
		t.Fatalf("tokPerSec(5ms) = %v, want 200", got)
	}
	if got := tokPerSec(0); got != 0 {
		t.Fatalf("tokPerSec(0) = %v, want 0", got)
	}
}

// TestSpineModelInfo_CopiesFields_Good proves the inference→spine model-info
// bridge carries the fields the durable session needs.
func TestSpineModelInfo_CopiesFields_Good(t *testing.T) {
	got := spineModelInfo(inference.ModelInfo{
		Architecture: "gemma4",
		VocabSize:    262144,
		NumLayers:    26,
		HiddenSize:   2304,
		QuantBits:    4,
		QuantGroup:   64,
	}, 8192)
	if got.Architecture != "gemma4" || got.VocabSize != 262144 || got.NumLayers != 26 ||
		got.HiddenSize != 2304 || got.QuantBits != 4 || got.QuantGroup != 64 {
		t.Fatalf("field mapping wrong: %+v", got)
	}
	if got.ContextLength != 8192 {
		t.Fatalf("ContextLength = %d, want 8192", got.ContextLength)
	}
}

// TestSpineModelInfo_DefaultContext_Bad proves a non-positive context length
// falls back to the 4096 default rather than producing a zero-length KV cache.
func TestSpineModelInfo_DefaultContext_Bad(t *testing.T) {
	if got := spineModelInfo(inference.ModelInfo{Architecture: "gemma4"}, 0); got.ContextLength != 4096 {
		t.Fatalf("default ContextLength = %d, want 4096", got.ContextLength)
	}
}

// TestResolvedDraftBlock_FlagWins_Good proves an explicit draft block overrides
// the engine default.
func TestResolvedDraftBlock_FlagWins_Good(t *testing.T) {
	if got := resolvedDraftBlock(7); got != 7 {
		t.Fatalf("resolvedDraftBlock(7) = %d, want 7", got)
	}
}

// TestResolvedDraftBlock_DefaultWhenZero_Bad proves a zero flag falls back to
// the shared MTP engine default.
func TestResolvedDraftBlock_DefaultWhenZero_Bad(t *testing.T) {
	if got := resolvedDraftBlock(0); got != serving.MTPDefaultDraftBlock {
		t.Fatalf("resolvedDraftBlock(0) = %d, want %d", got, serving.MTPDefaultDraftBlock)
	}
}
