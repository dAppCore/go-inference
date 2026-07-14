// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

// assistant_dflash_livetap_test.go proves the non-corrupting live aux tap two ways
// against the real engine, on both forward routes (recorded-ICB replay and
// re-encode):
//
//   - PARITY: ExtractAuxHiddensLive taps the SAME aux-layer hiddens for the
//     boundary token that ForwardCaptureHiddens returns for that token — byte
//     identical, because both surface the engine's own per-layer capture at the
//     same position.
//   - NON-CORRUPTION: a session that is tapped mid-decode keeps decoding
//     byte-identically to an untapped twin — the tap re-runs the boundary token's
//     forward idempotently, leaving pos and the KV cache unperturbed.

const (
	dfltDModel  = 256
	dfltHeads   = 4
	dfltKV      = 2
	dfltHeadDim = 64 // metallib-backed SDPA size
	dfltFF      = 512
	dfltVocab   = 32
	dfltLayers  = 3
	dfltMaxLen  = 32
)

// newLiveTapFixture builds a fresh synthetic-weight ArchSession (varied per-layer
// weights, so no op collapses to a vacuous constant) — the reusable target for the
// live-tap tests. Each call is an independent session with an empty cache.
func newLiveTapFixture(t testing.TB) func() *ArchSession {
	t.Helper()
	layers := make([]DecodeLayerWeights, dfltLayers)
	types := make([]string, dfltLayers)
	for li := range layers {
		layers[li] = forwardLayer(dfltDModel, dfltHeads, dfltKV, dfltHeadDim, dfltFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(dfltVocab*dfltDModel, 21))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dfltDModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dfltDModel, Heads: dfltHeads, KVHeads: dfltKV, HeadDim: dfltHeadDim, FF: dfltFF, Vocab: dfltVocab,
		GlobalHeadDim: dfltHeadDim, GlobalKVHeads: dfltKV,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: dfltHeadDim, RotaryDimLocal: dfltHeadDim, Layer: specs,
	}
	return func() *ArchSession {
		s, err := NewArchSession(g, arch, dfltMaxLen)
		if err != nil {
			t.Fatalf("NewArchSession: %v", err)
		}
		return s
	}
}

// decodeLive drives a fresh session over ids with the ordinary incremental decode
// step (stepID) — the live serving forward, building the KV cache token by token.
func decodeLive(t testing.TB, s *ArchSession, ids []int32) {
	t.Helper()
	for _, id := range ids {
		if _, err := s.stepID(id); err != nil {
			t.Fatalf("stepID(%d): %v", id, err)
		}
	}
}

// wantAuxRows runs ForwardCaptureHiddens over ids on a fresh session and returns
// the boundary (last) token's hidden at each aux layer — the reference the live
// tap must reproduce.
func wantAuxRows(t testing.TB, mk func() *ArchSession, ids []int32, auxLayers []int) [][]byte {
	t.Helper()
	_, perLayer, err := mk().ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens: %v", err)
	}
	rowBytes := dfltDModel * bf16Size
	last := len(ids) - 1
	want := make([][]byte, len(auxLayers))
	for i, layer := range auxLayers {
		want[i] = append([]byte(nil), perLayer[layer][last*rowBytes:(last+1)*rowBytes]...)
	}
	return want
}

// wantAuxRowsReencode is the same reference on the RE-ENCODE route: a fresh session
// forwards ids token by token through StepWithID (which drives stepToken under
// icbDisabledForTest) with the per-layer capture armed — exactly ForwardCaptureHiddens'
// non-ICB branch — and returns the boundary token's hidden at each aux layer. The
// caller must have icbDisabledForTest set.
func wantAuxRowsReencode(t testing.TB, mk func() *ArchSession, ids []int32, auxLayers []int) [][]byte {
	t.Helper()
	s := mk()
	defer func() { _ = s.Close() }()
	prevFlag, prevCap := captureLayerHiddens, capturedLayerHiddens
	captureLayerHiddens = true
	capturedLayerHiddens = nil
	defer func() { captureLayerHiddens = prevFlag; capturedLayerHiddens = prevCap }()
	for _, id := range ids {
		emb, err := s.embedID(id)
		if err != nil {
			t.Fatalf("embedID(%d): %v", id, err)
		}
		if _, err := s.StepWithID(id, emb); err != nil {
			t.Fatalf("StepWithID(%d): %v", id, err)
		}
	}
	N, T := dfltLayers, len(ids)
	if len(capturedLayerHiddens) != T*N {
		t.Fatalf("re-encode reference captured %d entries, want %d (T=%d × N=%d)", len(capturedLayerHiddens), T*N, T, N)
	}
	rowBytes := dfltDModel * bf16Size
	last := T - 1
	want := make([][]byte, len(auxLayers))
	for i, layer := range auxLayers {
		row := capturedLayerHiddens[last*N+layer]
		if len(row) != rowBytes {
			t.Fatalf("re-encode reference row for layer %d is %d bytes, want %d", layer, len(row), rowBytes)
		}
		want[i] = append([]byte(nil), row...)
	}
	return want
}

// TestExtractAuxHiddensLiveParityICB gates the live tap on the recorded-ICB replay
// path (the serving default for non-MoE targets): the tapped boundary hiddens are
// byte-identical to ForwardCaptureHiddens' capture of the same token.
func TestExtractAuxHiddensLiveParityICB(t *testing.T) {
	requireNativeRuntime(t)
	mk := newLiveTapFixture(t)
	ids := []int32{3, 7, 1, 9, 4}
	auxLayers := []int{0, 2}

	s := mk()
	defer func() { _ = s.Close() }()
	decodeLive(t, s, ids)

	got, err := s.ExtractAuxHiddensLive(ids[len(ids)-1], auxLayers)
	if err != nil {
		t.Fatalf("ExtractAuxHiddensLive: %v", err)
	}
	want := wantAuxRows(t, mk, ids, auxLayers)
	if len(got) != len(want) {
		t.Fatalf("tapped %d aux rows, want %d", len(got), len(want))
	}
	for i := range want {
		eqBytes(t, "ICB live tap aux hidden vs ForwardCaptureHiddens", got[i], want[i])
	}
}

// TestExtractAuxHiddensLiveParityReencode gates parity on the re-encode path
// (stepToken + captureLayerHiddens) by disabling the ICB replay. The reference
// must be captured on the SAME route: ForwardCaptureHiddens picks its route by
// s.state.icb != nil ALONE (it ignores icbDisabledForTest), so it would stay on
// the ICB route while this decode is re-encode — a cross-route comparison that
// diverges whenever ICB != re-encode for the model (metallib-dependent). In
// production the two never diverge (icbDisabledForTest is always false, so the tap
// and ForwardCaptureHiddens always agree on route); the flag is a test-only lever,
// so the reference here re-encodes too — a full-sequence stepToken capture.
func TestExtractAuxHiddensLiveParityReencode(t *testing.T) {
	requireNativeRuntime(t)
	prev := icbDisabledForTest
	icbDisabledForTest = true
	defer func() { icbDisabledForTest = prev }()

	mk := newLiveTapFixture(t)
	ids := []int32{2, 5, 8, 1}
	auxLayers := []int{1, 2}

	s := mk()
	defer func() { _ = s.Close() }()
	decodeLive(t, s, ids)

	got, err := s.ExtractAuxHiddensLive(ids[len(ids)-1], auxLayers)
	if err != nil {
		t.Fatalf("ExtractAuxHiddensLive: %v", err)
	}
	want := wantAuxRowsReencode(t, mk, ids, auxLayers)
	for i := range want {
		eqBytes(t, "re-encode live tap aux hidden vs re-encode capture", got[i], want[i])
	}
}

// TestExtractAuxHiddensLiveNonCorrupting is the load-bearing invariant: a session
// tapped mid-decode continues decoding byte-identically to an untapped twin, so
// the tap never perturbs the running serving cache.
func TestExtractAuxHiddensLiveNonCorrupting(t *testing.T) {
	requireNativeRuntime(t)
	mk := newLiveTapFixture(t)
	prompt := []int32{6, 2, 9}
	tail := []int32{4, 1, 7, 3} // decoded AFTER the tap fires

	// Reference twin: never tapped.
	ref := mk()
	defer func() { _ = ref.Close() }()
	decodeLive(t, ref, prompt)
	refTail := make([][]byte, len(tail))
	for i, id := range tail {
		h, err := ref.stepID(id)
		if err != nil {
			t.Fatalf("ref stepID(%d): %v", id, err)
		}
		refTail[i] = append([]byte(nil), h...)
	}

	// Tapped twin: extract aux hiddens at the prompt boundary, then decode the tail.
	tapped := mk()
	defer func() { _ = tapped.Close() }()
	decodeLive(t, tapped, prompt)
	if _, err := tapped.ExtractAuxHiddensLive(prompt[len(prompt)-1], []int{0, 1, 2}); err != nil {
		t.Fatalf("ExtractAuxHiddensLive: %v", err)
	}
	if tapped.Pos() != len(prompt) {
		t.Fatalf("tap moved pos to %d, want %d (tap must not advance the session)", tapped.Pos(), len(prompt))
	}
	for i, id := range tail {
		h, err := tapped.stepID(id)
		if err != nil {
			t.Fatalf("tapped stepID(%d): %v", id, err)
		}
		eqBytes(t, "post-tap decode hidden vs untapped twin", h, refTail[i])
	}
}

// TestExtractAuxHiddensLiveBad covers the honest refusals: no boundary token to tap
// (empty cache) and an aux layer outside the decoder stack.
func TestExtractAuxHiddensLiveBad(t *testing.T) {
	requireNativeRuntime(t)
	mk := newLiveTapFixture(t)

	empty := mk()
	defer func() { _ = empty.Close() }()
	if _, err := empty.ExtractAuxHiddensLive(1, []int{0}); err == nil {
		t.Fatal("expected error tapping an empty session (no boundary token), got nil")
	}

	s := mk()
	defer func() { _ = s.Close() }()
	decodeLive(t, s, []int32{1, 2})
	if _, err := s.ExtractAuxHiddensLive(2, []int{dfltLayers}); err == nil {
		t.Fatalf("expected error for out-of-range aux layer %d, got nil", dfltLayers)
	}
	if _, err := s.ExtractAuxHiddensLive(2, nil); err == nil {
		t.Fatal("expected error for empty aux-layer request, got nil")
	}
}
