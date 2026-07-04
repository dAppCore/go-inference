// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"math"
	"testing"

	"dappco.re/go/inference/safetensors"
)

func bf16T(vals []float32, shape ...int) safetensors.Tensor {
	data := make([]byte, len(vals)*2)
	for i, v := range vals {
		bits := math.Float32bits(v)
		r := uint16((bits + 0x7fff + ((bits >> 16) & 1)) >> 16)
		data[2*i], data[2*i+1] = byte(r), byte(r>>8)
	}
	return safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: data}
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

// mkHybridCheckpoint builds a synthetic 4-layer Qwen 3.6-shaped checkpoint: full_attention_interval 2 →
// layers 0,2 linear (gated-delta), 1,3 full attention. Untied lm_head.
func mkHybridCheckpoint() (map[string]safetensors.Tensor, []byte) {
	const D, vocab, FF, nLayers = 8, 32, 16, 4
	const VH, HD, convDim, K, vDim = 4, 8, 64, 4, 32 // gated-delta: KH=2,VH=4,HD=8 ⇒ vDim=32,qDim=16,convDim=64
	const AH, AKVH, AHD = 4, 2, 8                    // attention
	ts := map[string]safetensors.Tensor{
		"model.embed_tokens.weight": bf16T(syn(vocab*D, 1), vocab, D),
		"model.norm.weight":         bf16T(syn(D, 2), D),
		"lm_head.weight":            bf16T(syn(vocab*D, 3), vocab, D),
	}
	for i := 0; i < nLayers; i++ {
		lp := "model.layers." + itoa(i) + "."
		ts[lp+"input_layernorm.weight"] = bf16T(syn(D, i*100+1), D)
		ts[lp+"post_attention_layernorm.weight"] = bf16T(syn(D, i*100+2), D)
		ts[lp+"mlp.gate_proj.weight"] = bf16T(syn(FF*D, i*100+3), FF, D)
		ts[lp+"mlp.up_proj.weight"] = bf16T(syn(FF*D, i*100+4), FF, D)
		ts[lp+"mlp.down_proj.weight"] = bf16T(syn(D*FF, i*100+5), D, FF)
		if (i+1)%2 == 0 { // full attention
			ap := lp + "self_attn."
			ts[ap+"q_proj.weight"] = bf16T(syn(AH*AHD*D, i*100+10), AH*AHD, D)
			ts[ap+"k_proj.weight"] = bf16T(syn(AKVH*AHD*D, i*100+11), AKVH*AHD, D)
			ts[ap+"v_proj.weight"] = bf16T(syn(AKVH*AHD*D, i*100+12), AKVH*AHD, D)
			ts[ap+"o_proj.weight"] = bf16T(syn(D*AH*AHD, i*100+13), D, AH*AHD)
			ts[ap+"q_norm.weight"] = bf16T(syn(AHD, i*100+14), AHD)
			ts[ap+"k_norm.weight"] = bf16T(syn(AHD, i*100+15), AHD)
		} else { // linear (gated-delta)
			gp := lp + "linear_attn."
			ts[gp+"in_proj_qkv.weight"] = bf16T(syn(convDim*D, i*100+20), convDim, D)
			ts[gp+"conv1d.weight"] = bf16T(syn(convDim*K, i*100+21), convDim, 1, K)
			ts[gp+"conv1d.bias"] = bf16T(syn(convDim, i*100+22), convDim)
			ts[gp+"in_proj_a.weight"] = bf16T(syn(VH*D, i*100+23), VH, D)
			ts[gp+"A_log"] = bf16T(syn(VH, i*100+24), VH)
			ts[gp+"dt_bias"] = bf16T(syn(VH, i*100+25), VH)
			ts[gp+"in_proj_b.weight"] = bf16T(syn(VH*D, i*100+26), VH, D)
			ts[gp+"in_proj_z.weight"] = bf16T(syn(vDim*D, i*100+27), vDim, D)
			ts[gp+"norm.weight"] = bf16T(syn(HD, i*100+28), HD)
			ts[gp+"out_proj.weight"] = bf16T(syn(D*vDim, i*100+29), D, vDim)
		}
	}
	config := []byte(`{"hidden_size":8,"num_hidden_layers":4,"intermediate_size":16,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":32,"rms_norm_eps":1e-5,"rope_theta":1000000,"partial_rotary_factor":0.5,"full_attention_interval":2}`)
	return ts, config
}

// TestLoadComposedWrapperConfig covers the config branches the flat checkpoint never touches:
// the multimodal text_config nesting (effective()), rope_theta + partial_rotary_factor sourced
// from the nested rope_parameters object (the flat keys absent), the odd-rotary-dim rounding
// (0.5·headDim 6 = 3 → rounded down to 2), layer_types-driven full_attention dispatch, and the
// tied head (no lm_head → Output nil).
func TestLoadComposedWrapperConfig(t *testing.T) {
	const D, vocab, FF = 8, 32, 16
	const AH, AKVH, AHD = 4, 2, 6 // head_dim 6: 0.5·6 = 3, odd → the rd-- branch fires
	ts := map[string]safetensors.Tensor{
		"model.embed_tokens.weight": bf16T(syn(vocab*D, 1), vocab, D),
		"model.norm.weight":         bf16T(syn(D, 2), D),
		// no lm_head → tied
	}
	lp := "model.layers.0."
	ts[lp+"input_layernorm.weight"] = bf16T(syn(D, 11), D)
	ts[lp+"post_attention_layernorm.weight"] = bf16T(syn(D, 12), D)
	ts[lp+"mlp.gate_proj.weight"] = bf16T(syn(FF*D, 13), FF, D)
	ts[lp+"mlp.up_proj.weight"] = bf16T(syn(FF*D, 14), FF, D)
	ts[lp+"mlp.down_proj.weight"] = bf16T(syn(D*FF, 15), D, FF)
	ap := lp + "self_attn."
	ts[ap+"q_proj.weight"] = bf16T(syn(AH*AHD*D, 16), AH*AHD, D)
	ts[ap+"k_proj.weight"] = bf16T(syn(AKVH*AHD*D, 17), AKVH*AHD, D)
	ts[ap+"v_proj.weight"] = bf16T(syn(AKVH*AHD*D, 18), AKVH*AHD, D)
	ts[ap+"o_proj.weight"] = bf16T(syn(D*AH*AHD, 19), D, AH*AHD)
	ts[ap+"q_norm.weight"] = bf16T(syn(AHD, 20), AHD)
	ts[ap+"k_norm.weight"] = bf16T(syn(AHD, 21), AHD)

	config := []byte(`{"text_config":{"hidden_size":8,"num_hidden_layers":1,"intermediate_size":16,
		"num_attention_heads":4,"num_key_value_heads":2,"head_dim":6,"vocab_size":32,"rms_norm_eps":1e-5,
		"rope_parameters":{"rope_theta":500000,"partial_rotary_factor":0.5},
		"layer_types":["full_attention"]}}`)

	m, err := LoadComposed(ts, config)
	if err != nil {
		t.Fatalf("LoadComposed(wrapped config): %v", err)
	}
	if len(m.Layers) != 1 {
		t.Fatalf("layers = %d, want 1 (num_hidden_layers must come from text_config)", len(m.Layers))
	}
	if m.Output != nil {
		t.Fatal("no lm_head in the checkpoint → Output must be nil (tied)")
	}
	am, ok := m.Layers[0].Mixer.(*attnMixer)
	if !ok {
		t.Fatalf("layer 0 mixer is %T, want *attnMixer (layer_types full_attention dispatch)", m.Layers[0].Mixer)
	}
	if am.cfg.RopeTheta != 500000 {
		t.Fatalf("RopeTheta = %v, want 500000 (from the nested rope_parameters, no flat key)", am.cfg.RopeTheta)
	}
	if am.cfg.HeadDim != AHD || am.cfg.Heads != AH || am.cfg.KVHeads != AKVH {
		t.Fatalf("attention geometry = heads %d/kv %d/hd %d, want %d/%d/%d", am.cfg.Heads, am.cfg.KVHeads, am.cfg.HeadDim, AH, AKVH, AHD)
	}
	if am.cfg.RotaryDim != 2 {
		t.Fatalf("RotaryDim = %d, want 2 (0.5·head_dim 6 = 3, odd → rounded down)", am.cfg.RotaryDim)
	}
	if _, err := NewSession(m).Forward([]int32{1, 5}); err != nil {
		t.Fatalf("wrapped-config model forward: %v", err)
	}
	t.Log("wrapped config loaded: text_config + rope_parameters resolved, odd rotary dim rounded, tied head")
}

// TestLoadComposed loads the synthetic hybrid checkpoint, checks the per-layer dispatch is correct, the
// untied head is read, and the loaded model decodes end-to-end with decode==prefill.
func TestLoadComposed(t *testing.T) {
	ts, cfg := mkHybridCheckpoint()
	m, err := LoadComposed(ts, cfg)
	if err != nil {
		t.Fatalf("LoadComposed: %v", err)
	}
	if len(m.Layers) != 4 || m.D != 8 || m.Vocab != 32 {
		t.Fatalf("model dims wrong: layers=%d D=%d vocab=%d", len(m.Layers), m.D, m.Vocab)
	}
	if m.Output == nil {
		t.Error("lm_head present → Output should be untied, not nil")
	}
	want := []string{"gated_deltanet", "full_attention", "gated_deltanet", "full_attention"}
	for i, l := range m.Layers {
		if l.Mixer.Kind() != want[i] {
			t.Errorf("layer %d mixer kind %q, want %q (full_attention_interval dispatch)", i, l.Mixer.Kind(), want[i])
		}
	}

	tokens := []int32{1, 5, 9, 2, 7}
	prefill, err := NewSession(m).Forward(tokens)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	dec := NewSession(m)
	for t0, tok := range tokens {
		h, err := dec.Forward([]int32{tok})
		if err != nil {
			t.Fatalf("decode %d: %v", t0, err)
		}
		for i := 0; i < m.D; i++ {
			if h[i] != prefill[t0*m.D+i] {
				t.Fatalf("token %d hidden[%d] = %v != prefill %v", t0, i, h[i], prefill[t0*m.D+i])
			}
		}
	}
	gen, err := NewSession(m).Generate(tokens, 4, -1)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	t.Logf("loaded synthetic Qwen 3.6-shaped hybrid checkpoint: 4 layers (linear|full|linear|full), decodes end-to-end → %v", gen)
}
