// SPDX-Licence-Identifier: EUPL-1.2

package audio

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"slices"
	"testing"

	"dappco.re/go/inference/model"
)

// mask_golden_test.go pins the host Conformer validity-mask path (Mantis #1839, audio unit D), ported
// from engine/metal's audio_mask_golden_test.go so the shared host tower carries the same mask-threading
// pins as the native GPU-dispatch tower. The pure-logic pins prove the nil validity vector is byte-
// identical to the fully-valid (all-true) path, so a single fully-valid clip is unchanged end-to-end. The
// supervised golden then loads the real e2b-4bit tower via model.Load (engine-neutral, unlike metal's
// LoadTokenModelDir) and proves a PADDED clip (valid prefix + zeroed padding + a per-frame validity mask)
// matches the HF Gemma4AudioModel on its valid prefix and that the padding changes nothing downstream.

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

// TestBlockedMask_NilEqualsAllValid pins the validity-AND no-op: a nil validity vector and an all-true
// validity of length seqLen both produce the purely-positional blocked mask, so a fully-valid clip is
// byte-identical whether the caller threads a mask or not.
func TestBlockedMask_NilEqualsAllValid(t *testing.T) {
	const seqLen, chunk, past, future = 25, 12, 12, 0
	nB := (seqLen + chunk - 1) / chunk
	ctx := chunk + past + future
	allValid := make([]bool, seqLen)
	for i := range allValid {
		allValid[i] = true
	}
	nilMask := blockedMask(seqLen, nB, chunk, ctx, past, future, nil)
	allMask := blockedMask(seqLen, nB, chunk, ctx, past, future, allValid)
	if !slices.Equal(nilMask, allMask) {
		t.Fatal("blockedMask(nil) != blockedMask(allTrue) — the validity AND is not a no-op for a fully-valid clip")
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

// TestTower_Encode_MaskGolden threads a per-frame validity mask through the host tower for a PADDED clip
// and pins the valid prefix to HF, plus the end-to-end nil-vs-all-valid byte-identity. Skips without the
// local checkpoint (supervised parity, not a portable CI gate).
func TestTower_Encode_MaskGolden(t *testing.T) {
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

	lm, mapping, err := model.Load(dir)
	if err != nil {
		t.Fatalf("model.Load(%s): %v", dir, err)
	}
	if mapping != nil {
		defer func() { _ = mapping.Close() }()
	}
	if lm.Audio == nil {
		t.Fatal("model has no Conformer audio payload")
	}
	la := lm.Audio

	// Padded run: valid prefix + zeroed padding, with the per-frame validity mask.
	mask := make([]bool, g.TotalFrames)
	for i := range g.ValidFrames {
		mask[i] = true
	}
	towerPadded, err := Encode(melPadded, g.TotalFrames, g.MelBins, la, mask)
	if err != nil {
		t.Fatalf("Encode(padded, mask): %v", err)
	}
	validPrefix := towerPadded[:g.SeqValid*g.OutputDim]

	// (a) The valid prefix matches the HF padded-run valid prefix.
	assertCosineGE(t, "mask-valid-prefix-vs-HF", validPrefix, decodeF32LE(t, g.HFValidB64), 0.999)

	// Valid-only run: the same valid frames, no padding (a fully-valid clip).
	allValid := make([]bool, g.ValidFrames)
	for i := range allValid {
		allValid[i] = true
	}
	towerValidNil, err := Encode(melValid, g.ValidFrames, g.MelBins, la, nil)
	if err != nil {
		t.Fatalf("Encode(valid, nil): %v", err)
	}
	towerValidMask, err := Encode(melValid, g.ValidFrames, g.MelBins, la, allValid)
	if err != nil {
		t.Fatalf("Encode(valid, allTrue): %v", err)
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
