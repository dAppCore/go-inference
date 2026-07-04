// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/safetensors"
)

// BenchmarkHeadEncoderQuant is the balloon-gone counterpart of BenchmarkLMHeadQuant: it drives the
// RESIDENT 4-bit head (headEncoder.encode — the per-token serve head after split (d)) over a
// directory-loaded checkpoint, where the head weight is bound ONCE, not re-uploaded per token. The
// rss-grow-B/op metric is the tell — LMHeadQuant grows ~packed-weight-size per call (the ~503 MB
// tied embedding at 12B = the serve balloon), the resident head encoder keeps it FLAT (≈ one
// token's transient activation ÷ N). AX-11: synthetic, no model load.
func BenchmarkHeadEncoderQuant(b *testing.B) {
	if os.Getenv(MetallibPathEnv) == "" {
		b.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		b.Fatal(err)
	}
	const gs, bits = 64, 4
	// a vocab/dModel where the packed head is a few MB so a per-token re-upload would show clearly.
	cfg := g4.Config{
		HiddenSize: 2048, NumHiddenLayers: 1, IntermediateSize: 4096,
		NumAttentionHeads: 8, NumKeyValueHeads: 4, HeadDim: 256, VocabSize: 32768, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		b.Fatal(err)
	}
	ts := quantGemma4TensorsB(arch, gs, bits) // synthetic byte fills of the right sizes (no real quant needed)
	dir := b.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(mustEncodeB(ts))); err != nil {
		b.Fatal(err)
	}
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		b.Fatal(err)
	}
	defer dm.Close()
	sb, err := buildShardBuffers(dm)
	if err != nil {
		b.Fatal(err)
	}
	lm, err := model.Assemble(dm.Tensors, arch, model.StandardWeightNames())
	if err != nil {
		b.Fatal(err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		b.Fatal(err)
	}
	he, err := buildHeadEncoder(sb, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, arch.Hidden, arch.Vocab, gs, bits, arch.Eps, arch.SoftCap, true)
	if err != nil || he == nil {
		b.Fatalf("buildHeadEncoder: %v (nil=%v)", err, he == nil)
	}
	hidden := bf16ConstBytes(arch.Hidden, 0.01)
	packed := arch.Vocab * arch.Hidden * bits / 8
	b.Logf("packed head weight = %.1f MB (resident once, NOT re-uploaded per token)", float64(packed)/(1<<20))

	b.ResetTimer()
	rss0 := maxRSSBytes()
	for i := 0; i < b.N; i++ {
		if _, err := he.encode(hidden, false); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(maxRSSBytes()-rss0)/float64(b.N), "rss-grow-B/op")
}

// quantGemma4TensorsB / mustEncodeB are the *testing.B (no *testing.T) siblings used by the bench —
// synthetic byte fills of the correct sizes (the head encoder only maps + size-checks bytes, so an
// arbitrary pattern of the right length exercises the resident-weight path without real quantising).
func quantGemma4TensorsB(arch model.Arch, gs, bits int) map[string]safetensors.Tensor {
	ts := map[string]safetensors.Tensor{}
	n := byte(1)
	fill := func(sz int) []byte {
		d := make([]byte, sz)
		for j := range d {
			d[j] = n
		}
		n++
		return d
	}
	mkNorm := func(name string, elems int) {
		ts[name] = safetensors.Tensor{Dtype: "BF16", Shape: []int{elems}, Data: fill(elems * bf16Size)}
	}
	mkQuant := func(prefix string, outDim, inDim int) {
		ts[prefix+".weight"] = safetensors.Tensor{Dtype: "U32", Shape: []int{outDim, inDim * bits / 32}, Data: fill(outDim * inDim * bits / 8)}
		ts[prefix+".scales"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / gs}, Data: fill(outDim * (inDim / gs) * bf16Size)}
		ts[prefix+".biases"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / gs}, Data: fill(outDim * (inDim / gs) * bf16Size)}
	}
	dModel, headDim, dFF, vocab := arch.Hidden, arch.HeadDim, arch.FF, arch.Vocab
	qDim, kvDim := arch.Heads*headDim, arch.KVHeads*headDim
	mkQuant("model.embed_tokens", vocab, dModel)
	mkNorm("model.norm.weight", dModel)
	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		mkNorm(p+".input_layernorm.weight", dModel)
		mkNorm(p+".pre_feedforward_layernorm.weight", dModel)
		mkNorm(p+".self_attn.q_norm.weight", headDim)
		mkNorm(p+".self_attn.k_norm.weight", headDim)
		mkNorm(p+".post_attention_layernorm.weight", dModel)
		mkNorm(p+".post_feedforward_layernorm.weight", dModel)
		mkQuant(p+".self_attn.q_proj", qDim, dModel)
		mkQuant(p+".self_attn.k_proj", kvDim, dModel)
		mkQuant(p+".self_attn.v_proj", kvDim, dModel)
		mkQuant(p+".self_attn.o_proj", dModel, qDim)
		mkQuant(p+".mlp.gate_proj", dFF, dModel)
		mkQuant(p+".mlp.up_proj", dFF, dModel)
		mkQuant(p+".mlp.down_proj", dModel, dFF)
	}
	return ts
}

func mustEncodeB(ts map[string]safetensors.Tensor) []byte {
	b, _ := safetensors.Encode(ts)
	return b
}
