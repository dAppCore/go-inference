// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/safetensors"
)

// benchQuantProj builds a projection-sized 8-bit packed weight (with its .scales/.biases siblings) and a
// matching quant config — the input tensorAsQuant reads. 8-bit avoids the b1→b2 repack so the copy-vs-view
// difference is exactly the packed-weight copy.
func benchQuantProj(outDim, inDim, bits, gs int) (map[string]safetensors.Tensor, string, *model.QuantConfig) {
	w := make([]float32, outDim*inDim)
	for i := range w {
		w[i] = float32((i%13)-6) * 0.05
	}
	packed, scales, biases, err := mlxaffine.QuantizeTensor(w, outDim, inDim, bits, gs)
	if err != nil {
		panic(err)
	}
	name := "model.layers.0.mlp.gate_proj.weight"
	base := name[:len(name)-len(".weight")]
	ts := map[string]safetensors.Tensor{
		name:             {Dtype: "U32", Shape: []int{outDim, mlxaffine.PackedWords(inDim, bits)}, Data: packed},
		base + ".scales": {Dtype: "BF16", Shape: []int{outDim, inDim / gs}, Data: scales},
		base + ".biases": {Dtype: "BF16", Shape: []int{outDim, inDim / gs}, Data: biases},
	}
	return ts, name, &model.QuantConfig{GroupSize: gs, Bits: bits}
}

// BenchmarkTensorAsQuant_Copy — the owned-copy path (LoadComposed): the packed weight is heap-copied, so
// B/op carries the whole packed-weight allocation (plus the small scales/biases copies).
func BenchmarkTensorAsQuant_Copy(b *testing.B) {
	ts, name, q := benchQuantProj(2048, 2048, 8, 64)
	t := ts[name]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := tensorAsQuant(ts, name, t, q, false); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkTensorAsQuant_ZeroCopy — the view path (LoadComposedDir): the packed weight ALIASES the input
// bytes (no allocation), so its contribution to B/op is 0; only the small bf16 scales/biases copies remain.
// The B/op delta from _Copy is exactly the eliminated packed-weight heap copy — the RSS win.
func BenchmarkTensorAsQuant_ZeroCopy(b *testing.B) {
	ts, name, q := benchQuantProj(2048, 2048, 8, 64)
	t := ts[name]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := tensorAsQuant(ts, name, t, q, true); err != nil {
			b.Fatal(err)
		}
	}
}

// benchMoEQuantCheckpoint builds a small qwen2_moe-shaped checkpoint in memory — every layer full attention
// + a 4-expert MoE FFN — with its attention/embed/lm_head projections quantised (8-bit, gs 8, no repack) and
// its experts left bf16. It is the input the MoE load-path benches feed to LoadComposedWithArch (copy) vs
// LoadComposedWithArchMmap (zero-copy); the B/op delta is exactly the eliminated packed-projection copies.
func benchMoEQuantCheckpoint() (map[string]safetensors.Tensor, []byte, model.Arch) {
	const D, vocab, nLayers = 8, 32, 2
	const AH, AKVH, AHD, nE, moeFF, bits, gs = 4, 2, 8, 4, 16, 8, 8
	quant := func(rows, cols int, seed int) map[string]safetensors.Tensor { // packed U32 weight + bf16 sidecars
		w := make([]float32, rows*cols)
		for i := range w {
			w[i] = float32((i+seed)%13-6) * 0.05
		}
		packed, scales, biases, err := mlxaffine.QuantizeTensor(w, rows, cols, bits, gs)
		if err != nil {
			panic(err)
		}
		return map[string]safetensors.Tensor{
			".weight": {Dtype: "U32", Shape: []int{rows, mlxaffine.PackedWords(cols, bits)}, Data: packed},
			".scales": {Dtype: "BF16", Shape: []int{rows, cols / gs}, Data: scales},
			".biases": {Dtype: "BF16", Shape: []int{rows, cols / gs}, Data: biases},
		}
	}
	put := func(ts map[string]safetensors.Tensor, base string, q map[string]safetensors.Tensor) {
		for suffix, tensor := range q {
			ts[base+suffix] = tensor
		}
	}
	ts := map[string]safetensors.Tensor{"model.norm.weight": bf16T(syn(D, 2), D)}
	put(ts, "model.embed_tokens", quant(vocab, D, 1))
	put(ts, "lm_head", quant(vocab, D, 3))
	for i := range nLayers {
		lp := "model.layers." + itoa(i) + "."
		ts[lp+"input_layernorm.weight"] = bf16T(syn(D, i*100+1), D)
		ts[lp+"post_attention_layernorm.weight"] = bf16T(syn(D, i*100+2), D)
		ap := lp + "self_attn."
		put(ts, ap+"q_proj", quant(AH*AHD, D, i*100+10))
		put(ts, ap+"k_proj", quant(AKVH*AHD, D, i*100+11))
		put(ts, ap+"v_proj", quant(AKVH*AHD, D, i*100+12))
		put(ts, ap+"o_proj", quant(D, AH*AHD, i*100+13))
		mp := lp + "mlp."
		ts[mp+"gate.weight"] = bf16T(syn(nE*D, i*100+20), nE, D)
		for e := range nE {
			ep := mp + "experts." + itoa(e) + "."
			ts[ep+"gate_proj.weight"] = bf16T(syn(moeFF*D, i*100+e*3+30), moeFF, D)
			ts[ep+"up_proj.weight"] = bf16T(syn(moeFF*D, i*100+e*3+31), moeFF, D)
			ts[ep+"down_proj.weight"] = bf16T(syn(D*moeFF, i*100+e*3+32), D, moeFF)
		}
	}
	cfg := []byte(`{"model_type":"qwen2_moe","hidden_size":8,"num_hidden_layers":2,"intermediate_size":16,` +
		`"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":32,"rms_norm_eps":1e-5,` +
		`"rope_theta":1000000,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":16,` +
		`"quantization":{"group_size":8,"bits":8}}`)
	arch := model.Arch{Experts: nE, TopK: 2, MoEGating: model.MoEGatingSoftmax, EmbedScale: 1}
	return ts, cfg, arch
}

// BenchmarkLoadComposedWithArch_Copy — the owned-copy MoE build (LoadComposedWithArch): every packed
// projection weight is heap-copied, so B/op carries all those packed-weight allocations.
func BenchmarkLoadComposedWithArch_Copy(b *testing.B) {
	ts, cfg, arch := benchMoEQuantCheckpoint()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := LoadComposedWithArch(ts, cfg, arch); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkLoadComposedWithArchMmap_ZeroCopy — the zero-copy MoE build (LoadComposedWithArchMmap, the path
// the MoE registry hooks now use): the packed projection weights ALIAS the input bytes, so their copies drop
// out of B/op. The B/op delta from _Copy is the eliminated packed-projection heap copies — the MoE RSS win.
func BenchmarkLoadComposedWithArchMmap_ZeroCopy(b *testing.B) {
	ts, cfg, arch := benchMoEQuantCheckpoint()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := LoadComposedWithArchMmap(ts, cfg, arch); err != nil {
			b.Fatal(err)
		}
	}
}

// The loader benches baseline the per-load widening + layer-schedule work (AX-11):
// tensorF32 widens a bf16 checkpoint tensor to a flat f32 slice — run once per weight at
// load, its allocation is the whole f32 copy (the biggest single load allocation, sized to
// the weight). resolveKinds maps each layer to full/linear attention from the config, a
// small []string per load. Neither is per-token; these pin the load-time cost. Synthetic
// tensors — no checkpoint read.

// BenchmarkTensorF32_BF16 — widening a projection-sized bf16 weight to f32: the per-element
// left-shift unpack into a fresh [len/2] f32 buffer. The dominant per-weight load allocation.
func BenchmarkTensorF32_BF16(b *testing.B) {
	n := benchFF * benchD
	t := safetensors.Tensor{Shape: []int{benchFF, benchD}, Dtype: "BF16", Data: make([]byte, n*2)}
	b.SetBytes(int64(len(t.Data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := tensorF32(t); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkResolveKinds_Interval — deriving a 48-layer schedule from full_attention_interval
// (every 6th layer full): one []string allocation + a modulo pass. The per-load layer-typing.
func BenchmarkResolveKinds_Interval(b *testing.B) {
	cfg := &loaderConfig{NumHiddenLayers: 48, FullAttentionInterval: 6}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := resolveKinds(cfg); err != nil {
			b.Fatal(err)
		}
	}
}
