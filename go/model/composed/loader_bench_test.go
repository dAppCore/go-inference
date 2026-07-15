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
