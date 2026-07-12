// SPDX-Licence-Identifier: EUPL-1.2

package generate

import (
	"context"
	"testing"
	"time"
	"unicode/utf8"

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

// TestKVStorageEncoding pins the flag → kv.Encoding resolution: native/q8/float32
// are recognised (all produce-able from a live -state sleep now that the block
// capture emits per-head float32 for non-native encodings); an unknown value is
// not recognised and falls back to native.
func TestKVStorageEncoding(t *testing.T) {
	cases := []struct {
		raw        string
		enc        kv.Encoding
		recognised bool
	}{
		{"", kv.EncodingNative, true},
		{"native", kv.EncodingNative, true},
		{"Q8", kv.EncodingQ8, true},
		{"float32", kv.KVSnapshotEncodingFloat32, true},
		{"fp16", kv.EncodingNative, false}, // go-mlx-era vocabulary, not a real encoding here
	}
	for _, c := range cases {
		enc, recognised := kvStorageEncoding(c.raw)
		if enc != c.enc || recognised != c.recognised {
			t.Fatalf("kvStorageEncoding(%q) = (%q, %v), want (%q, %v)",
				c.raw, enc, recognised, c.enc, c.recognised)
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

// stubModelPath is any non-empty path — the -state multimodal guards reject
// before RunGenerate reaches a model load, so this path is never opened.
const stubModelPath = "/nonexistent/model/dir"

// TestRunGenerate_StateImageRejected_Bad proves an image turn combined with
// -state is rejected before any load — the durable session prefills text only.
// It also drives the -native notice and the -context-len load-option threading
// on the way in (both run before the guard fires), so the pure entry path is
// covered without touching a GPU.
func TestRunGenerate_StateImageRejected_Bad(t *testing.T) {
	log := core.NewBuffer()
	err := RunGenerate(context.Background(), Config{
		StateName:    "s",
		ModelPath:    stubModelPath,
		ImageSources: []string{"data:image/png;base64,AAAA"},
		Native:       true,
		ContextLen:   2048,
		Out:          core.NewBuffer(),
		Log:          log,
	})
	if err == nil {
		t.Fatal("image input with -state: want rejection, got nil")
	}
	if !core.Contains(err.Error(), "image") {
		t.Fatalf("error %q, want it to name image input", err.Error())
	}
	if !core.Contains(log.String(), "native") {
		t.Fatalf("log %q, want the -native notice", log.String())
	}
}

// TestRunGenerate_StateAudioRejected_Bad proves an audio turn with -state is
// rejected before any load (audio is a stateless-only input).
func TestRunGenerate_StateAudioRejected_Bad(t *testing.T) {
	err := RunGenerate(context.Background(), Config{
		StateName:    "s",
		ModelPath:    stubModelPath,
		AudioSources: []string{"data:audio/wav;base64,AAAA"},
	})
	if err == nil {
		t.Fatal("audio input with -state: want rejection, got nil")
	}
	if !core.Contains(err.Error(), "audio") {
		t.Fatalf("error %q, want it to name audio input", err.Error())
	}
}

// TestRunGenerate_StateVideoRejected_Ugly proves a video-frame turn with -state
// is rejected before any load (video is a stateless-only input).
func TestRunGenerate_StateVideoRejected_Ugly(t *testing.T) {
	err := RunGenerate(context.Background(), Config{
		StateName:         "s",
		ModelPath:         stubModelPath,
		VideoFrameSources: []string{"data:image/png;base64,AAAA"},
	})
	if err == nil {
		t.Fatal("video input with -state: want rejection, got nil")
	}
	if !core.Contains(err.Error(), "video") {
		t.Fatalf("error %q, want it to name video input", err.Error())
	}
}

// TestLoadTextModel_MissingDir_Bad proves loadTextModel surfaces an error (and a
// nil model) when the metal backend cannot load the path. In the portable test
// binary the "metal" backend is unregistered, so LoadModel fails fast; the arm
// under test is the failure-to-model handling, not the GPU load itself.
func TestLoadTextModel_MissingDir_Bad(t *testing.T) {
	tm, err := loadTextModel(core.PathJoin(t.TempDir(), "no-such-model"))
	if err == nil {
		if tm != nil {
			tm.Close()
		}
		t.Fatal("loadTextModel of a missing dir: want error, got nil")
	}
	if tm != nil {
		t.Fatalf("loadTextModel returned a non-nil model alongside an error: %v", tm)
	}
}

// TestPrintNote_NilWriter_Silent_Bad proves printNote is a no-op with a nil
// writer — RunGenerate passes a nil Log on the quiet paths, so a notice must
// silently drop rather than panic.
func TestPrintNote_NilWriter_Silent_Bad(t *testing.T) {
	printNote(nil, "this should be swallowed %d", 1) // must not panic
}

// fakeMTPModel adds the optional speculative-metrics capability so
// printMTPMetrics can be exercised without arming a real drafter.
type fakeMTPModel struct {
	inference.TextModel
	metrics inference.SpeculativeMetrics
}

func (f fakeMTPModel) SpeculativeMetrics() inference.SpeculativeMetrics { return f.metrics }

// Compile-time proof the fake carries the interface printMTPMetrics asserts.
var _ inference.SpeculativeMetricsProvider = fakeMTPModel{}

// TestPrintMTPMetrics_NoProvider_Bad proves a model that exposes no speculative
// seam prints nothing — the MTP line is silent unless speculation engaged.
func TestPrintMTPMetrics_NoProvider_Bad(t *testing.T) {
	out := core.NewBuffer()
	printMTPMetrics(out, fakeTextModel{})
	if s := out.String(); s != "" {
		t.Fatalf("non-speculative model printed %q, want silence", s)
	}
}

// TestPrintMTPMetrics_ZeroProposed_Ugly proves a provider that reports no
// proposed tokens (speculation never engaged) stays silent.
func TestPrintMTPMetrics_ZeroProposed_Ugly(t *testing.T) {
	out := core.NewBuffer()
	printMTPMetrics(out, fakeMTPModel{})
	if s := out.String(); s != "" {
		t.Fatalf("zero-proposed metrics printed %q, want silence", s)
	}
}

// TestPrintMTPMetrics_Accepted_Good proves an engaged speculative lane renders
// the acceptance line the bench reads to judge whether the drafter earns its keep.
func TestPrintMTPMetrics_Accepted_Good(t *testing.T) {
	out := core.NewBuffer()
	printMTPMetrics(out, fakeMTPModel{metrics: inference.SpeculativeMetrics{
		ProposedTokens:    10,
		AcceptedTokens:    7,
		AcceptanceRate:    0.7,
		TargetVerifyCalls: 4,
	}})
	s := out.String()
	for _, want := range []string{"mtp:", "70%", "7/10", "4 verify"} {
		if !core.Contains(s, want) {
			t.Fatalf("mtp line %q missing %q", s, want)
		}
	}
}

// TestWarmPrefix_ShortPassthrough_Good pins the common case: a prompt at or
// under the bound is returned unchanged, so short-prompt warms behave exactly
// as before the bound existed.
func TestWarmPrefix_ShortPassthrough_Good(t *testing.T) {
	if got := warmPrefix("hello"); got != "hello" {
		t.Fatalf("warmPrefix(short) = %q, want passthrough", got)
	}
	exact := make([]byte, warmPrefixChars)
	for i := range exact {
		exact[i] = 'a'
	}
	if got := warmPrefix(string(exact)); len(got) != warmPrefixChars {
		t.Fatalf("warmPrefix(exact) len = %d, want %d", len(got), warmPrefixChars)
	}
}

// TestWarmPrefix_TruncatesLong_Bad pins the deep-prompt case that motivated the
// bound: a prompt far past warmPrefixChars comes back bounded, so the warm pass
// never replays the full prefill.
func TestWarmPrefix_TruncatesLong_Bad(t *testing.T) {
	long := make([]byte, warmPrefixChars*4)
	for i := range long {
		long[i] = 'b'
	}
	got := warmPrefix(string(long))
	if len(got) != warmPrefixChars {
		t.Fatalf("warmPrefix(long) len = %d, want %d", len(got), warmPrefixChars)
	}
}

// TestWarmPrefix_RuneBoundary_Ugly pins the truncation backing off to a rune
// boundary: a multi-byte rune straddling the cut is dropped whole rather than
// split into an invalid UTF-8 tail.
func TestWarmPrefix_RuneBoundary_Ugly(t *testing.T) {
	head := make([]byte, warmPrefixChars-1)
	for i := range head {
		head[i] = 'c'
	}
	s := string(head) + "€€€" // 3-byte runes straddle the cut at warmPrefixChars
	got := warmPrefix(s)
	if len(got) > warmPrefixChars {
		t.Fatalf("warmPrefix over bound: %d", len(got))
	}
	if !utf8.ValidString(got) {
		t.Fatalf("warmPrefix split a rune: tail %q", got[len(got)-4:])
	}
}
