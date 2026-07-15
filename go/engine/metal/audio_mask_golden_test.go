// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"slices"
	"testing"
)

// audio_mask_golden_test.go pins the Conformer validity-mask path (Mantis #1839, audio unit D). The pure
// -logic pins prove the nil validity vector is byte-identical to the fully-valid (all-true) path, so a
// single fully-valid clip is unchanged end-to-end. The supervised golden then loads the real e2b-4bit
// tower and proves a PADDED clip (valid prefix + zeroed padding + a per-frame validity mask) matches the
// HF Gemma4AudioModel on its valid prefix and that the padding changes nothing downstream — the exact
// behaviour that DIVERGED before the mask was threaded (a positional-only blocked mask attended padding).

// TestHalveValidity_Good pins HF's mask[:, ::2]: the stride-2 subsample keeps every other entry and the
// halved length equals the conv's output time length convOut, so two halvings track the two convs.
func TestHalveValidity_Good(t *testing.T) {
	got := halveValidity([]bool{true, true, false, true, false}) // keep idx 0,2,4
	if want := []bool{true, false, false}; !slices.Equal(got, want) {
		t.Fatalf("halveValidity = %v, want %v", got, want)
	}
	if halveValidity(nil) != nil {
		t.Fatal("halveValidity(nil) must stay nil (a fully-valid clip)")
	}
	for _, n := range []int{1, 30, 31, 32, 33, 64} {
		if l := len(halveValidity(make([]bool, n))); l != convOut(n) {
			t.Fatalf("len(halveValidity(%d))=%d, want convOut=%d — mask must track the conv time axis", n, l, convOut(n))
		}
	}
}

// TestAudioBlockedMask_NilEqualsAllValid pins the validity-AND no-op: a nil validity vector and an
// all-true validity of length seqLen both produce the purely-positional blocked mask, so a fully-valid
// clip is byte-identical whether the caller threads a mask or not.
func TestAudioBlockedMask_NilEqualsAllValid(t *testing.T) {
	const seqLen, chunk, past, future = 25, 12, 12, 0
	nB := (seqLen + chunk - 1) / chunk
	ctx := chunk + past + future
	allValid := make([]bool, seqLen)
	for i := range allValid {
		allValid[i] = true
	}
	nilMask := audioBlockedMask(seqLen, nB, chunk, ctx, past, future, nil)
	allMask := audioBlockedMask(seqLen, nB, chunk, ctx, past, future, allValid)
	if !slices.Equal(nilMask, allMask) {
		t.Fatal("audioBlockedMask(nil) != audioBlockedMask(allTrue) — the validity AND is not a no-op for a fully-valid clip")
	}
}

type audioMaskGolden struct {
	ValidFrames   int    `json:"valid_frames"`
	TotalFrames   int    `json:"total_frames"`
	MelBins       int    `json:"mel_bins"`
	SeqValid      int    `json:"seq_valid"`
	SeqTotal      int    `json:"seq_total"`
	OutputDim     int    `json:"output_dim"`
	MelPaddedB64  string `json:"mel_padded_bf16le_b64"`
	HFValidB64    string `json:"hf_valid_prefix_f32le_b64"`
	HFUnpaddedB64 string `json:"hf_unpadded_f32le_b64"`
}

// TestAudioTowerMaskGolden threads a per-frame validity mask through the native tower for a PADDED clip
// and pins the valid prefix to HF, plus the end-to-end nil-vs-all-valid byte-identity. Skips without the
// local checkpoint (supervised parity, not a portable CI gate).
//
// The fixture uses valid_frames=32 (⇒ ceil(32/2)=16, EVEN): the two stride-2 subsample convs keep the
// halved-mask boundary off every one of the 8 valid soft-tokens, so the valid prefix matches HF to the
// tower's GPU-GEMM rounding floor. When ceil(valid_frames/2) is ODD the last valid soft-token reads the
// first halved-mask-false layer0 frame through the layer1 conv: HF zeros that frame between the two convs
// (Gemma4AudioSubSampleConvProjectionLayer's `hidden_states * mask`), the native subsampler does not (it
// only halves the mask — see AudioSubsampleF32), so that one token diverges (~cosine 0.998, max|Δ|~5).
// Closing that residual needs the inter-layer r0 zeroing, which shifts single-clip boundary soft-tokens
// and is therefore out of this "halve + attention-AND" unit's byte-identity scope.
func TestAudioTowerMaskGolden(t *testing.T) {
	dir := e2b4bitSnapshotDir()
	if dir == "" {
		t.Skip("e2b-it-4bit checkpoint not in HF cache — supervised parity test")
	}
	raw, err := os.ReadFile("testdata/audio_mask_golden.json")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g audioMaskGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("unmarshal golden: %v", err)
	}
	melPadded, err := base64.StdEncoding.DecodeString(g.MelPaddedB64)
	if err != nil {
		t.Fatalf("mel b64: %v", err)
	}
	if len(melPadded) != g.TotalFrames*g.MelBins*2 {
		t.Fatalf("mel bytes %d want %d", len(melPadded), g.TotalFrames*g.MelBins*2)
	}
	melValid := melPadded[:g.ValidFrames*g.MelBins*2]

	tm, err := LoadTokenModelDir(dir, 2048)
	if err != nil {
		t.Fatalf("LoadTokenModelDir(%s): %v", dir, err)
	}
	if c, ok := tm.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	m, ok := tm.(*NativeTokenModel)
	if !ok {
		t.Fatalf("token model is %T, want *NativeTokenModel", tm)
	}
	if m.audio == nil {
		t.Fatal("model has no Conformer audio payload")
	}

	// Padded run: valid prefix + zeroed padding, with the per-frame validity mask.
	weightsF, cfg, _, ok := nativeAudioFromLoaded(m.audio, g.TotalFrames, g.MelBins)
	if !ok {
		t.Fatal("nativeAudioFromLoaded(total) failed")
	}
	mask := make([]bool, g.TotalFrames)
	for i := range g.ValidFrames {
		mask[i] = true
	}
	towerPadded, err := AudioEncode(melPadded, weightsF, cfg, mask)
	if err != nil {
		t.Fatalf("AudioEncode(padded, mask): %v", err)
	}
	validPrefix := towerPadded[:g.SeqValid*g.OutputDim]

	// (a) The valid prefix matches the HF padded-run valid prefix.
	assertCosineGE(t, "mask-valid-prefix-vs-HF", validPrefix, decodeF32LE(t, g.HFValidB64), 0.999)

	// Valid-only run: the same valid frames, no padding (a fully-valid clip).
	weightsV, _, _, ok := nativeAudioFromLoaded(m.audio, g.ValidFrames, g.MelBins)
	if !ok {
		t.Fatal("nativeAudioFromLoaded(valid) failed")
	}
	allValid := make([]bool, g.ValidFrames)
	for i := range allValid {
		allValid[i] = true
	}
	towerValidNil, err := AudioEncode(melValid, weightsV, cfg, nil)
	if err != nil {
		t.Fatalf("AudioEncode(valid, nil): %v", err)
	}
	towerValidMask, err := AudioEncode(melValid, weightsV, cfg, allValid)
	if err != nil {
		t.Fatalf("AudioEncode(valid, allTrue): %v", err)
	}

	// (b) nil vs all-valid mask is byte-identical end-to-end (the fully-valid clip is unchanged).
	if d := maxAbsDelta(towerValidNil, towerValidMask); d != 0 {
		t.Fatalf("nil vs all-valid tower max|Δ|=%g, want 0 — the mask path is not byte-identical for a fully-valid clip", d)
	}
	t.Logf("nil-vs-all-valid byte-identity: max|Δ|=0 (n=%d)", len(towerValidNil))

	// (c) Padding changes nothing downstream: the padded run's valid prefix equals the unpadded run.
	assertCosineGE(t, "padded-prefix-vs-unpadded", validPrefix, towerValidNil, 0.9999)

	// (d) Anchor: the unpadded run matches HF unpadded (the standard module-golden check on the clip).
	assertCosineGE(t, "unpadded-vs-HF", towerValidNil, decodeF32LE(t, g.HFUnpaddedB64), 0.999)
}
