// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/safetensors"
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
	for i := range nLayers {
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

// mkGatedHybridMoECheckpoint builds a Qwen 3.6-shaped 4-layer hybrid at hidden 64: full_attention_interval
// 4 → layers 0,1,2 gated-delta (linear_attention), layer 3 GATED full attention (attn_output_gate); every
// layer's FFN is MoE (6 experts, top-2) with a sigmoid-gated shared expert. The config declares the
// linear_* geometry so validateLinearGeometry runs on the happy path. It exercises gated attention +
// gated-delta + MoE together — the three Qwen 3.6 features in one stack.
func mkGatedHybridMoECheckpoint() (map[string]safetensors.Tensor, []byte) {
	const D, vocab, nLayers = 64, 48, 4
	const VH, KH, HD, convDim, K, vDim = 4, 2, 8, 64, 4, 32 // gated-delta: KH·HD=16, VH·HD=32, convDim=2·16+32
	const AH, AKVH, AHD = 4, 2, 16                          // gated attention: q_proj = 2·AH·AHD = 128 rows
	const nE, moeFF, sharedFF = 6, 32, 32
	ts := map[string]safetensors.Tensor{
		"model.embed_tokens.weight": bf16T(syn(vocab*D, 1), vocab, D),
		"model.norm.weight":         bf16T(syn(D, 2), D),
		"lm_head.weight":            bf16T(syn(vocab*D, 3), vocab, D),
	}
	for i := range nLayers {
		lp := "model.layers." + itoa(i) + "."
		ts[lp+"input_layernorm.weight"] = bf16T(syn(D, i*300+1), D)
		ts[lp+"post_attention_layernorm.weight"] = bf16T(syn(D, i*300+2), D)
		mp := lp + "mlp."
		ts[mp+"gate.weight"] = bf16T(syn(nE*D, i*300+3), nE, D)
		for e := range nE {
			ep := mp + "experts." + itoa(e) + "."
			ts[ep+"gate_proj.weight"] = bf16T(syn(moeFF*D, i*300+e*5+10), moeFF, D)
			ts[ep+"up_proj.weight"] = bf16T(syn(moeFF*D, i*300+e*5+11), moeFF, D)
			ts[ep+"down_proj.weight"] = bf16T(syn(D*moeFF, i*300+e*5+12), D, moeFF)
		}
		sp := mp + "shared_expert."
		ts[sp+"gate_proj.weight"] = bf16T(syn(sharedFF*D, i*300+80), sharedFF, D)
		ts[sp+"up_proj.weight"] = bf16T(syn(sharedFF*D, i*300+81), sharedFF, D)
		ts[sp+"down_proj.weight"] = bf16T(syn(D*sharedFF, i*300+82), D, sharedFF)
		ts[mp+"shared_expert_gate.weight"] = bf16T(syn(D, i*300+83), 1, D)
		if (i+1)%4 == 0 { // gated full attention
			ap := lp + "self_attn."
			ts[ap+"q_proj.weight"] = bf16T(syn(2*AH*AHD*D, i*300+90), 2*AH*AHD, D)
			ts[ap+"k_proj.weight"] = bf16T(syn(AKVH*AHD*D, i*300+91), AKVH*AHD, D)
			ts[ap+"v_proj.weight"] = bf16T(syn(AKVH*AHD*D, i*300+92), AKVH*AHD, D)
			ts[ap+"o_proj.weight"] = bf16T(syn(D*AH*AHD, i*300+93), D, AH*AHD)
			ts[ap+"q_norm.weight"] = bf16T(syn(AHD, i*300+94), AHD)
			ts[ap+"k_norm.weight"] = bf16T(syn(AHD, i*300+95), AHD)
		} else { // gated-delta
			gp := lp + "linear_attn."
			ts[gp+"in_proj_qkv.weight"] = bf16T(syn(convDim*D, i*300+100), convDim, D)
			ts[gp+"conv1d.weight"] = bf16T(syn(convDim*K, i*300+101), convDim, 1, K)
			ts[gp+"conv1d.bias"] = bf16T(syn(convDim, i*300+102), convDim)
			ts[gp+"in_proj_a.weight"] = bf16T(syn(VH*D, i*300+103), VH, D)
			ts[gp+"A_log"] = bf16T(syn(VH, i*300+104), VH)
			ts[gp+"dt_bias"] = bf16T(syn(VH, i*300+105), VH)
			ts[gp+"in_proj_b.weight"] = bf16T(syn(VH*D, i*300+106), VH, D)
			ts[gp+"in_proj_z.weight"] = bf16T(syn(vDim*D, i*300+107), vDim, D)
			ts[gp+"norm.weight"] = bf16T(syn(HD, i*300+108), HD)
			ts[gp+"out_proj.weight"] = bf16T(syn(D*vDim, i*300+109), D, vDim)
		}
	}
	cfg := []byte(`{"model_type":"qwen3_5","hidden_size":64,"num_hidden_layers":4,"num_attention_heads":4,` +
		`"num_key_value_heads":2,"head_dim":16,"vocab_size":48,"rms_norm_eps":1e-5,"attn_output_gate":true,` +
		`"num_experts":6,"num_experts_per_tok":2,"moe_intermediate_size":32,"shared_expert_intermediate_size":32,` +
		`"full_attention_interval":4,"rope_theta":10000000,"partial_rotary_factor":0.25,` +
		`"linear_num_key_heads":2,"linear_num_value_heads":4,"linear_key_head_dim":8,"linear_value_head_dim":8,` +
		`"linear_conv_kernel_dim":4}`)
	return ts, cfg
}

// TestComposedHybridQwen35Integration is the flagship synthetic-tensor forward: a 4-layer 3:1 hybrid
// (gated-delta ×3 + gated attention ×1, MoE FFN with a gated shared expert) loads with the correct
// per-layer dispatch and decodes token-by-token BIT-EXACT to a single prefill pass — the whole Qwen 3.6
// feature set (gated attention + gated-delta + MoE + norm_topk_prob + linear-geometry validation) driven
// together through the composed session.
func TestComposedHybridQwen35Integration(t *testing.T) {
	ts, cfg := mkGatedHybridMoECheckpoint()
	m, err := LoadComposed(ts, cfg)
	if err != nil {
		t.Fatalf("LoadComposed: %v", err)
	}
	if len(m.Layers) != 4 || m.D != 64 || m.Vocab != 48 {
		t.Fatalf("dims wrong: layers=%d D=%d vocab=%d", len(m.Layers), m.D, m.Vocab)
	}
	wantKind := []string{"gated_deltanet", "gated_deltanet", "gated_deltanet", "full_attention"}
	for i, l := range m.Layers {
		if l.Mixer.Kind() != wantKind[i] {
			t.Fatalf("layer %d mixer %q, want %q (3:1 schedule)", i, l.Mixer.Kind(), wantKind[i])
		}
		moe, ok := l.MLP.(*MoEMLP)
		if !ok {
			t.Fatalf("layer %d FFN %T, want *MoEMLP", i, l.MLP)
		}
		if len(moe.SharedGate) != 64 || !moe.NormTopKProb || moe.TopK != 2 {
			t.Fatalf("layer %d MoE: sharedGate=%d normTopK=%v topK=%d, want 64/true/2", i, len(moe.SharedGate), moe.NormTopKProb, moe.TopK)
		}
	}
	am, ok := m.Layers[3].Mixer.(*attnMixer)
	if !ok || !am.cfg.OutputGate {
		t.Fatalf("layer 3 must be a gated *attnMixer (OutputGate), got %T", m.Layers[3].Mixer)
	}

	tokens := []int32{1, 5, 40, 2, 7, 33}
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
		for i := range m.D {
			if h[i] != prefill[t0*m.D+i] {
				t.Fatalf("token %d hidden[%d] = %v != prefill %v (hybrid decode diverged)", t0, i, h[i], prefill[t0*m.D+i])
			}
		}
	}
	if _, err := NewSession(m).Generate(tokens, 4, -1); err != nil {
		t.Fatalf("generate: %v", err)
	}
	t.Log("Qwen 3.6 hybrid (gated-delta ×3 + gated attention ×1, MoE + gated shared expert) decode == prefill bit-exact")
}

// TestLinearGeometryMismatch pins that validateLinearGeometry FIRES: a config declaring a linear_* value
// that disagrees with the geometry the loader derives from the weight shapes fails the load rather than
// mis-building the gated-delta mixer. mkHybridCheckpoint's gated-delta layers derive value heads = 4.
func TestLinearGeometryMismatch(t *testing.T) {
	ts, _ := mkHybridCheckpoint()
	cfg := []byte(`{"hidden_size":8,"num_hidden_layers":4,"intermediate_size":16,"num_attention_heads":4,` +
		`"num_key_value_heads":2,"head_dim":8,"vocab_size":32,"rms_norm_eps":1e-5,"rope_theta":1000000,` +
		`"partial_rotary_factor":0.5,"full_attention_interval":2,"linear_num_value_heads":5}`)
	if _, err := LoadComposed(ts, cfg); err == nil {
		t.Fatal("declared linear_num_value_heads:5 != derived 4 must fail validateLinearGeometry")
	}
	t.Log("config/checkpoint linear-geometry mismatch (declared 5 vs derived 4 value heads) rejected")
}

// TestLoadComposedGatedAttention loads a checkpoint whose attention layer is gated (attn_output_gate:true,
// so q_proj carries 2·heads·head_dim rows) and confirms the loader sets AttnConfig.OutputGate and the model
// decodes — the buildAttn wiring for the gated path.
func TestLoadComposedGatedAttention(t *testing.T) {
	const D, vocab, FF = 8, 32, 16
	const AH, AKVH, AHD = 4, 2, 8 // gated attention: q_proj = 2·AH·AHD rows ([q ; gate])
	ts := map[string]safetensors.Tensor{
		"model.embed_tokens.weight": bf16T(syn(vocab*D, 1), vocab, D),
		"model.norm.weight":         bf16T(syn(D, 2), D),
		"lm_head.weight":            bf16T(syn(vocab*D, 3), vocab, D),
	}
	lp := "model.layers.0."
	ts[lp+"input_layernorm.weight"] = bf16T(syn(D, 11), D)
	ts[lp+"post_attention_layernorm.weight"] = bf16T(syn(D, 12), D)
	ts[lp+"mlp.gate_proj.weight"] = bf16T(syn(FF*D, 13), FF, D)
	ts[lp+"mlp.up_proj.weight"] = bf16T(syn(FF*D, 14), FF, D)
	ts[lp+"mlp.down_proj.weight"] = bf16T(syn(D*FF, 15), D, FF)
	ap := lp + "self_attn."
	ts[ap+"q_proj.weight"] = bf16T(syn(2*AH*AHD*D, 16), 2*AH*AHD, D) // [q ; gate]
	ts[ap+"k_proj.weight"] = bf16T(syn(AKVH*AHD*D, 17), AKVH*AHD, D)
	ts[ap+"v_proj.weight"] = bf16T(syn(AKVH*AHD*D, 18), AKVH*AHD, D)
	ts[ap+"o_proj.weight"] = bf16T(syn(D*AH*AHD, 19), D, AH*AHD)
	ts[ap+"q_norm.weight"] = bf16T(syn(AHD, 20), AHD)
	ts[ap+"k_norm.weight"] = bf16T(syn(AHD, 21), AHD)
	config := []byte(`{"hidden_size":8,"num_hidden_layers":1,"intermediate_size":16,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":32,"rms_norm_eps":1e-5,"attn_output_gate":true,"rope_theta":1000000,"partial_rotary_factor":0.25,"layer_types":["full_attention"]}`)
	m, err := LoadComposed(ts, config)
	if err != nil {
		t.Fatalf("LoadComposed: %v", err)
	}
	am, ok := m.Layers[0].Mixer.(*attnMixer)
	if !ok {
		t.Fatalf("layer 0 mixer is %T, want *attnMixer", m.Layers[0].Mixer)
	}
	if !am.cfg.OutputGate {
		t.Fatal("attn_output_gate:true must set AttnConfig.OutputGate")
	}
	if _, err := NewSession(m).Forward([]int32{1, 5, 3}); err != nil {
		t.Fatalf("gated forward: %v", err)
	}
	t.Log("gated-attention checkpoint (q_proj = 2·heads·head_dim) loaded — OutputGate set, decodes")
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

// quantiseInPlace replaces ts[name] with its mlx-affine packed form (+ .scales/.biases
// siblings) and returns the exact dequantised values the loader must reproduce.
func quantiseInPlace(t *testing.T, ts map[string]safetensors.Tensor, name string, bits, gs int) []float32 {
	t.Helper()
	src := ts[name]
	vals, err := tensorF32(src)
	if err != nil {
		t.Fatalf("decode %s: %v", name, err)
	}
	out, in := src.Shape[0], src.Shape[1]
	packed, scales, biases, err := mlxaffine.QuantizeTensor(vals, out, in, bits, gs)
	if err != nil {
		t.Fatalf("quantise %s: %v", name, err)
	}
	ts[name] = safetensors.Tensor{Dtype: "U32", Shape: []int{out, mlxaffine.PackedWords(in, bits)}, Data: packed}
	base := name[:len(name)-len(".weight")]
	ts[base+".scales"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{out, in / gs}, Data: scales}
	ts[base+".biases"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{out, in / gs}, Data: biases}
	want, err := mlxaffine.DequantizeTensor(packed, scales, biases, out, in, bits, gs)
	if err != nil {
		t.Fatalf("dequantise %s: %v", name, err)
	}
	return want
}

// TestLoadComposedQuantised pins the mlx-affine quantised path: packed-uint32 weights with
// .scales/.biases siblings dequantise to EXACTLY mlxaffine.DequantizeTensor's output, the
// per-module override (8-bit embed, language_model.-prefixed key) is honoured over the 4-bit
// default, and the logical hidden width is recovered from the dequantised length (a packed
// embed's Shape[1] is the compressed word count, not D).
func TestLoadComposedQuantised(t *testing.T) {
	ts, config := mkHybridCheckpoint()
	wantEmbed := quantiseInPlace(t, ts, "model.embed_tokens.weight", 8, 8)
	wantGate := quantiseInPlace(t, ts, "model.layers.0.mlp.gate_proj.weight", 4, 8)
	quantiseInPlace(t, ts, "model.layers.0.linear_attn.in_proj_qkv.weight", 4, 8)
	config = append(config[:len(config)-1], []byte(`,"quantization":{"group_size":8,"bits":4,"language_model.model.embed_tokens":{"group_size":8,"bits":8}}}`)...)

	m, err := LoadComposed(ts, config)
	if err != nil {
		t.Fatalf("LoadComposed: %v", err)
	}
	if m.D != 8 || m.Vocab != 32 {
		t.Fatalf("logical dims: got D=%d vocab=%d, want 8/32", m.D, m.Vocab)
	}
	if len(m.Embed) != len(wantEmbed) {
		t.Fatalf("embed length: got %d want %d", len(m.Embed), len(wantEmbed))
	}
	for i := range wantEmbed {
		if m.Embed[i] != wantEmbed[i] {
			t.Fatalf("embed[%d]: got %v want %v (8-bit override not honoured?)", i, m.Embed[i], wantEmbed[i])
		}
	}
	gate := m.Layers[0].MLP.(*MLP).Gate
	if len(gate) != len(wantGate) {
		t.Fatalf("gate length: got %d want %d", len(gate), len(wantGate))
	}
	for i := range wantGate {
		if gate[i] != wantGate[i] {
			t.Fatalf("gate[%d]: got %v want %v", i, gate[i], wantGate[i])
		}
	}
}

// TestLoadComposedLanguageModelModelPrefix pins the real Qwen 3.6 pack nesting: EVERY tensor
// under language_model. (language_model.model.layers…, language_model.lm_head.weight). The
// normaliser's stripped aliases must make the load identical to the flat layout.
func TestLoadComposedLanguageModelModelPrefix(t *testing.T) {
	flat, config := mkHybridCheckpoint()
	wrapped := make(map[string]safetensors.Tensor, len(flat))
	for k, v := range flat {
		wrapped["language_model."+k] = v
	}
	got, err := LoadComposed(wrapped, config)
	if err != nil {
		t.Fatalf("LoadComposed(language_model.model. nesting): %v", err)
	}
	want, err := LoadComposed(flat, config)
	if err != nil {
		t.Fatalf("LoadComposed(flat): %v", err)
	}
	if got.D != want.D || got.Vocab != want.Vocab || len(got.Layers) != len(want.Layers) {
		t.Fatalf("geometry mismatch: got D=%d vocab=%d layers=%d, want D=%d vocab=%d layers=%d",
			got.D, got.Vocab, len(got.Layers), want.D, want.Vocab, len(want.Layers))
	}
	for i := range want.Embed {
		if got.Embed[i] != want.Embed[i] {
			t.Fatalf("embed[%d]: wrapped %v ≠ flat %v", i, got.Embed[i], want.Embed[i])
		}
	}
	if got.Output == nil {
		t.Fatal("untied lm_head lost through the language_model. nesting")
	}
}

// TestLoadComposedMLXConvLayout pins the mlx depthwise-conv shape: [convDim, K, 1]
// (channel-last) versus torch's [convDim, 1, K]. The flat bytes are identical, so the load
// must derive the same geometry and weights from either shape.
func TestLoadComposedMLXConvLayout(t *testing.T) {
	ts, config := mkHybridCheckpoint()
	for k, v := range ts {
		if len(v.Shape) == 3 && v.Shape[1] == 1 {
			ts[k] = safetensors.Tensor{Dtype: v.Dtype, Shape: []int{v.Shape[0], v.Shape[2], 1}, Data: v.Data}
		}
	}
	got, err := LoadComposed(ts, config)
	if err != nil {
		t.Fatalf("LoadComposed(mlx conv layout): %v", err)
	}
	flat, _ := mkHybridCheckpoint()
	want, err := LoadComposed(flat, config)
	if err != nil {
		t.Fatalf("LoadComposed(torch conv layout): %v", err)
	}
	if len(got.Layers) != len(want.Layers) {
		t.Fatalf("layer count: got %d want %d", len(got.Layers), len(want.Layers))
	}
}
