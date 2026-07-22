// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// bf16Tensor builds a bf16 safetensors.Tensor from f32 values with the given shape.
func bf16Tensor(vals []float32, shape ...int) safetensors.Tensor {
	data := make([]byte, len(vals)*2)
	for i, v := range vals {
		bits := math.Float32bits(v)
		r := uint16((bits + 0x7fff + ((bits >> 16) & 1)) >> 16)
		data[2*i], data[2*i+1] = byte(r), byte(r>>8)
	}
	return safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: data}
}

// TestLoader_LoadMambaModel_Good builds a synthetic 2-layer Mamba-2 checkpoint (the standard HF
// names/shapes), loads it, and verifies the geometry is derived correctly from the weight shapes and
// the loaded model runs an end-to-end recurrent decode.
func TestLoader_LoadMambaModel_Good(t *testing.T) {
	const H, headDim, N, K = 2, 8, 8, 4
	const dInner = H * headDim         // 16
	const convDim = dInner + 2*N       // 32
	const projOut = 2*dInner + 2*N + H // 50
	const D, vocab, nLayers = 8, 32, 2

	ts := map[string]safetensors.Tensor{
		"backbone.embeddings.weight": bf16Tensor(syn(vocab*D, 1), vocab, D),
		"backbone.norm_f.weight":     bf16Tensor(syn(D, 2), D),
	}
	for li := range nLayers {
		mp := "backbone.layers." + itoa(li) + ".mixer."
		ts["backbone.layers."+itoa(li)+".norm.weight"] = bf16Tensor(syn(D, li*7+3), D)
		ts[mp+"in_proj.weight"] = bf16Tensor(syn(projOut*D, li*7+4), projOut, D)
		ts[mp+"conv1d.weight"] = bf16Tensor(syn(convDim*K, li*7+5), convDim, 1, K)
		ts[mp+"conv1d.bias"] = bf16Tensor(syn(convDim, li*7+6), convDim)
		ts[mp+"A_log"] = bf16Tensor(syn(H, li*7+7), H)
		ts[mp+"D"] = bf16Tensor(syn(H, li*7+8), H)
		ts[mp+"dt_bias"] = bf16Tensor(syn(H, li*7+9), H)
		ts[mp+"norm.weight"] = bf16Tensor(syn(dInner, li*7+10), dInner)
		ts[mp+"out_proj.weight"] = bf16Tensor(syn(D*dInner, li*7+11), D, dInner)
	}

	m, err := LoadMambaModel(ts, 1e-5)
	if err != nil {
		t.Fatalf("LoadMambaModel: %v", err)
	}
	want := BlockConfig{NumHeads: H, HeadDim: headDim, StateDim: N, NumGroups: 1, ConvKernel: K, Eps: 1e-5}
	if m.Cfg != want {
		t.Fatalf("derived geometry %+v, want %+v", m.Cfg, want)
	}
	if m.D != D || m.Vocab != vocab || len(m.Layers) != nLayers {
		t.Fatalf("model dims wrong: D=%d vocab=%d layers=%d", m.D, m.Vocab, len(m.Layers))
	}
	if m.LMHead != nil {
		t.Error("LMHead should be nil (tied) — no lm_head.weight in the checkpoint")
	}
	gen, err := NewSession(m).Generate([]int32{1, 2, 3}, 4, -1)
	if err != nil {
		t.Fatalf("Generate on loaded model: %v", err)
	}
	if len(gen) != 4 {
		t.Fatalf("generated %d tokens, want 4", len(gen))
	}
	t.Logf("loaded synthetic Mamba-2 checkpoint: geometry %+v from shapes, %d layers, decodes end-to-end → %v", m.Cfg, len(m.Layers), gen)
}

func TestLoader_LoadMambaModel_Bad(t *testing.T) {
	if _, err := LoadMambaModel(map[string]safetensors.Tensor{}, 1e-5); err == nil {
		t.Fatal("empty checkpoint accepted")
	}
}

// TestLoader_LoadMambaModel_Ugly proves the per-layer-uniform-geometry assumption is enforced: a
// checkpoint whose layer 1 declares a different head count than layer 0 is rejected, not silently
// accepted with layer 0's geometry — distinct from _Bad's totally-empty checkpoint.
func TestLoader_LoadMambaModel_Ugly(t *testing.T) {
	const headDim, N, K = 8, 8, 4
	const D = 8

	ts := map[string]safetensors.Tensor{
		"backbone.embeddings.weight": bf16Tensor(syn(32*D, 1), 32, D),
		"backbone.norm_f.weight":     bf16Tensor(syn(D, 2), D),
	}
	layer := func(li, h int) {
		dInner := h * headDim
		convDim := dInner + 2*N
		projOut := 2*dInner + 2*N + h
		mp := "backbone.layers." + itoa(li) + ".mixer."
		ts["backbone.layers."+itoa(li)+".norm.weight"] = bf16Tensor(syn(D, li*7+3), D)
		ts[mp+"in_proj.weight"] = bf16Tensor(syn(projOut*D, li*7+4), projOut, D)
		ts[mp+"conv1d.weight"] = bf16Tensor(syn(convDim*K, li*7+5), convDim, 1, K)
		ts[mp+"A_log"] = bf16Tensor(syn(h, li*7+7), h)
		ts[mp+"out_proj.weight"] = bf16Tensor(syn(D*dInner, li*7+11), D, dInner)
	}
	layer(0, 2) // layer 0: H=2
	layer(1, 4) // layer 1: H=4 — geometry mismatch, must be rejected

	if _, err := LoadMambaModel(ts, 1e-5); err == nil {
		t.Fatal("layer 1 geometry differing from layer 0 accepted")
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [20]byte
	p := len(b)
	for i > 0 {
		p--
		b[p] = byte('0' + i%10)
		i /= 10
	}
	return string(b[p:])
}
