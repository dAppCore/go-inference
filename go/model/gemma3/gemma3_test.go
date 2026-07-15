// SPDX-Licence-Identifier: EUPL-1.2

package gemma3

import (
	"math"
	"testing"

	"dappco.re/go/inference/model"
)

// TestGemma3Arch verifies the gemma3 config→Arch derivation against the specifics confirmed from metal
// gemma3: scale 1/sqrt(head_dim), full rotary, no softcap, no value-norm, and the sliding/global layer
// pattern (global when (i+1)%pattern == 0).
func TestGemma3Arch(t *testing.T) {
	const layers, pattern, headDim = 12, 6, 256
	c := &Config{
		HiddenSize: 2048, NumHiddenLayers: layers, IntermediateSize: 8192,
		NumAttentionHeads: 8, NumKeyValueHeads: 4, HeadDim: headDim, VocabSize: 262144,
		RMSNormEps: 1e-6, RopeTheta: 1_000_000, RopeLocalBaseFreq: 10_000,
		SlidingWindow: 1024, SlidingWindowPattern: pattern,
	}
	a, err := c.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if want := float32(1.0 / math.Sqrt(headDim)); a.AttnScale != want {
		t.Errorf("AttnScale = %v, want 1/sqrt(head_dim) = %v", a.AttnScale, want)
	}
	if a.RotaryDim != headDim || a.RotaryDimLocal != headDim {
		t.Errorf("rotary = %d/%d, want full %d (gemma3 has no partial rotary)", a.RotaryDim, a.RotaryDimLocal, headDim)
	}
	if a.SoftCap != 0 {
		t.Errorf("SoftCap = %v, want 0 (gemma3 dropped logit softcapping)", a.SoftCap)
	}
	if a.ValueNorm {
		t.Error("ValueNorm = true, want false (gemma3 does not value-norm V)")
	}
	if a.RopeBase != 1_000_000 || a.RopeLocalBase != 10_000 {
		t.Errorf("rope bases = %v/%v, want 1e6/1e4", a.RopeBase, a.RopeLocalBase)
	}
	if a.Hidden != 2048 || a.Heads != 8 || a.KVHeads != 4 || a.HeadDim != headDim || a.FF != 8192 || a.Vocab != 262144 {
		t.Errorf("dims wrong: %+v", a)
	}
	if len(a.Layer) != layers {
		t.Fatalf("layers = %d, want %d", len(a.Layer), layers)
	}
	globals := 0
	for i := range a.Layer {
		isGlobal := a.Layer[i].Attention == model.GlobalAttention
		wantGlobal := (i+1)%pattern == 0
		if isGlobal != wantGlobal {
			t.Errorf("layer %d: global=%v, want %v", i, isGlobal, wantGlobal)
		}
		if isGlobal {
			globals++
		}
	}
	t.Logf("gemma3 Arch: scale=1/sqrt(%d), full rotary, no softcap/value-norm, %d layers (%d global per pattern %d)", headDim, layers, globals, pattern)
}

// TestGemma3Registered confirms gemma3 is in the reactive arch registry with the gemma (1+w) RMSNorm
// convention enabled (NormBiasOne) — so model.Load assembles a gemma3 checkpoint with folded norms.
func TestGemma3Registered(t *testing.T) {
	var spec model.ArchSpec
	for _, mt := range []string{"gemma3", "gemma3_text"} { // both declared aliases must resolve
		s, ok := model.LookupArch(mt)
		if !ok {
			t.Fatalf("gemma3 not registered in the arch registry under %q", mt)
		}
		spec = s
	}
	if !spec.Weights.NormBiasOne {
		t.Error("gemma3 ArchSpec must set Weights.NormBiasOne for the (1+w) RMSNorm convention")
	}
	// Parse a minimal config and derive the arch through the registered spec.
	cfg, err := spec.Parse([]byte(`{"model_type":"gemma3","hidden_size":1152,"num_hidden_layers":4,"intermediate_size":6912,"num_attention_heads":4,"num_key_value_heads":1,"head_dim":256,"vocab_size":262144,"sliding_window_pattern":6}`))
	if err != nil {
		t.Fatalf("registered Parse: %v", err)
	}
	a, err := cfg.Arch()
	if err != nil {
		t.Fatalf("registered Arch: %v", err)
	}
	if len(a.Layer) != 4 || a.HeadDim != 256 {
		t.Fatalf("parsed arch wrong: layers=%d headDim=%d", len(a.Layer), a.HeadDim)
	}
	t.Log("gemma3 registered: model.Load can parse + assemble a gemma3 checkpoint via the reactive loader")
}
