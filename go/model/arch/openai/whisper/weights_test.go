// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// f32Tensor builds an F32 safetensors.Tensor from f32 values (mirrors mamba2's bf16Tensor test helper —
// arch/mamba2/loader_test.go — but F32, matching every published Whisper checkpoint's torch_dtype).
func f32Tensor(vals []float32, shape ...int) safetensors.Tensor {
	data := make([]byte, len(vals)*4)
	for i, v := range vals {
		bits := math.Float32bits(v)
		data[4*i] = byte(bits)
		data[4*i+1] = byte(bits >> 8)
		data[4*i+2] = byte(bits >> 16)
		data[4*i+3] = byte(bits >> 24)
	}
	return safetensors.Tensor{Dtype: "F32", Shape: shape, Data: data}
}

// seqVals returns [0, 1, ..., n-1] as f32 — a value that trivially proves round-trip identity (each
// loaded element equals its own index) without needing a real checkpoint.
func seqVals(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(i)
	}
	return v
}

// tinyWhisperTensors builds a hermetic 1-encoder/1-decoder-layer checkpoint using Whisper's REAL tensor
// names (not simplified — the "never guessed" precedent gptossSyntheticTensors sets for its family),
// small enough to hand-construct: d=4, mel=2, ffn=6, vocab=5, maxSrc=3, maxTgt=3.
func tinyWhisperTensors() (map[string]safetensors.Tensor, *Config) {
	const d, mel, ffn, vocab, maxSrc, maxTgt = 4, 2, 6, 5, 3, 3
	tensors := map[string]safetensors.Tensor{
		"model.encoder.conv1.weight":           f32Tensor(seqVals(d*mel*3), d, mel, 3),
		"model.encoder.conv1.bias":             f32Tensor(seqVals(d), d),
		"model.encoder.conv2.weight":           f32Tensor(seqVals(d*d*3), d, d, 3),
		"model.encoder.conv2.bias":             f32Tensor(seqVals(d), d),
		"model.encoder.embed_positions.weight": f32Tensor(seqVals(maxSrc*d), maxSrc, d),
		"model.encoder.layer_norm.weight":      f32Tensor(seqVals(d), d),
		"model.encoder.layer_norm.bias":        f32Tensor(seqVals(d), d),

		"model.decoder.embed_tokens.weight":    f32Tensor(seqVals(vocab*d), vocab, d),
		"model.decoder.embed_positions.weight": f32Tensor(seqVals(maxTgt*d), maxTgt, d),
		"model.decoder.layer_norm.weight":      f32Tensor(seqVals(d), d),
		"model.decoder.layer_norm.bias":        f32Tensor(seqVals(d), d),
	}
	for _, p := range []string{"model.encoder.layers.0", "model.decoder.layers.0"} {
		tensors[p+".self_attn_layer_norm.weight"] = f32Tensor(seqVals(d), d)
		tensors[p+".self_attn_layer_norm.bias"] = f32Tensor(seqVals(d), d)
		tensors[p+".self_attn.q_proj.weight"] = f32Tensor(seqVals(d*d), d, d)
		tensors[p+".self_attn.q_proj.bias"] = f32Tensor(seqVals(d), d)
		tensors[p+".self_attn.k_proj.weight"] = f32Tensor(seqVals(d*d), d, d) // NO bias — Whisper's k_proj
		tensors[p+".self_attn.v_proj.weight"] = f32Tensor(seqVals(d*d), d, d)
		tensors[p+".self_attn.v_proj.bias"] = f32Tensor(seqVals(d), d)
		tensors[p+".self_attn.out_proj.weight"] = f32Tensor(seqVals(d*d), d, d)
		tensors[p+".self_attn.out_proj.bias"] = f32Tensor(seqVals(d), d)
		tensors[p+".final_layer_norm.weight"] = f32Tensor(seqVals(d), d)
		tensors[p+".final_layer_norm.bias"] = f32Tensor(seqVals(d), d)
		tensors[p+".fc1.weight"] = f32Tensor(seqVals(ffn*d), ffn, d)
		tensors[p+".fc1.bias"] = f32Tensor(seqVals(ffn), ffn)
		tensors[p+".fc2.weight"] = f32Tensor(seqVals(d*ffn), d, ffn)
		tensors[p+".fc2.bias"] = f32Tensor(seqVals(d), d)
	}
	// decoder layer 0 also carries cross-attention
	tensors["model.decoder.layers.0.encoder_attn_layer_norm.weight"] = f32Tensor(seqVals(d), d)
	tensors["model.decoder.layers.0.encoder_attn_layer_norm.bias"] = f32Tensor(seqVals(d), d)
	tensors["model.decoder.layers.0.encoder_attn.q_proj.weight"] = f32Tensor(seqVals(d*d), d, d)
	tensors["model.decoder.layers.0.encoder_attn.q_proj.bias"] = f32Tensor(seqVals(d), d)
	tensors["model.decoder.layers.0.encoder_attn.k_proj.weight"] = f32Tensor(seqVals(d*d), d, d)
	tensors["model.decoder.layers.0.encoder_attn.v_proj.weight"] = f32Tensor(seqVals(d*d), d, d)
	tensors["model.decoder.layers.0.encoder_attn.v_proj.bias"] = f32Tensor(seqVals(d), d)
	tensors["model.decoder.layers.0.encoder_attn.out_proj.weight"] = f32Tensor(seqVals(d*d), d, d)
	tensors["model.decoder.layers.0.encoder_attn.out_proj.bias"] = f32Tensor(seqVals(d), d)

	cfg := &Config{
		DModel: d, NumMelBins: mel, EncoderFFNDim: ffn, DecoderFFNDim: ffn, VocabSize: vocab,
		MaxSourcePositions: maxSrc, MaxTargetPositions: maxTgt,
		EncoderLayers: 1, DecoderLayers: 1, EncoderAttentionHeads: 2, DecoderAttentionHeads: 2,
	}
	return tensors, cfg
}

func TestLoadWeights_Good(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	w, err := LoadWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	if len(w.EncoderLayers) != 1 || len(w.DecoderLayers) != 1 {
		t.Fatalf("loaded %d encoder / %d decoder layers, want 1/1", len(w.EncoderLayers), len(w.DecoderLayers))
	}
	if w.DModel != 4 || w.VocabSize != 5 || w.MaxSourcePositions != 3 || w.MaxTargetPositions != 3 {
		t.Fatalf("geometry = %+v, want DModel 4/Vocab 5/MaxSrc 3/MaxTgt 3", w)
	}
	// round-trip identity: seqVals means loaded[i] == i exactly (f32 represents small integers exactly).
	for i, v := range w.Conv1Bias {
		if v != float32(i) {
			t.Fatalf("Conv1Bias[%d] = %v, want %v (round-trip identity)", i, v, i)
		}
	}
	if w.EncoderLayers[0].SelfAttn.K.Bias != nil {
		t.Fatal("encoder self_attn.k_proj must load with NO bias (Whisper's WhisperAttention: k_proj bias=False)")
	}
	if w.DecoderLayers[0].SelfAttn.K.Bias != nil {
		t.Fatal("decoder self_attn.k_proj must load with NO bias")
	}
	if w.DecoderLayers[0].CrossAttn.K.Bias != nil {
		t.Fatal("decoder cross-attn (encoder_attn) k_proj must load with NO bias")
	}
	if w.DecoderLayers[0].CrossAttn.Q.Bias == nil || w.DecoderLayers[0].CrossAttn.V.Bias == nil || w.DecoderLayers[0].CrossAttn.Out.Bias == nil {
		t.Fatal("decoder cross-attn q/v/out_proj must all carry bias")
	}
}

func TestLoadWeights_Bad(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	delete(tensors, "model.encoder.conv1.weight")
	if _, err := LoadWeights(tensors, cfg); err == nil {
		t.Fatal("LoadWeights accepted a checkpoint missing model.encoder.conv1.weight")
	}
}

// TestLoadWeights_Ugly proves a present-but-wrong-shaped tensor is refused (not silently misread) —
// distinct from _Bad's missing-tensor case.
func TestLoadWeights_Ugly(t *testing.T) {
	tensors, cfg := tinyWhisperTensors()
	tensors["model.encoder.conv1.bias"] = f32Tensor(seqVals(999), 999) // wrong length for DModel=4
	if _, err := LoadWeights(tensors, cfg); err == nil {
		t.Fatal("LoadWeights accepted a wrong-shaped conv1.bias")
	}
}

func TestLoadWeights_NilConfig_Bad(t *testing.T) {
	tensors, _ := tinyWhisperTensors()
	if _, err := LoadWeights(tensors, nil); err == nil {
		t.Fatal("LoadWeights accepted a nil config")
	}
}

func TestLoadWeights_IncompleteGeometry_Bad(t *testing.T) {
	tensors, _ := tinyWhisperTensors()
	if _, err := LoadWeights(tensors, &Config{}); err == nil {
		t.Fatal("LoadWeights accepted a zero-value config")
	}
}

func TestTensorF32_Good(t *testing.T) {
	got, err := tensorF32(f32Tensor([]float32{1.5, -2.25, 3}, 3))
	if err != nil {
		t.Fatalf("tensorF32: %v", err)
	}
	want := []float32{1.5, -2.25, 3}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("tensorF32 = %v, want %v", got, want)
		}
	}
}

func TestTensorF32_Bad(t *testing.T) {
	if _, err := tensorF32(safetensors.Tensor{Dtype: "U8", Data: []byte{1, 2, 3}}); err == nil {
		t.Fatal("tensorF32 accepted an unsupported dtype")
	}
}

// TestTensorF32_Ugly proves a truncated (misaligned) byte length is refused rather than silently
// truncating the last partial element.
func TestTensorF32_Ugly(t *testing.T) {
	if _, err := tensorF32(safetensors.Tensor{Dtype: "F32", Data: []byte{1, 2, 3}}); err == nil {
		t.Fatal("tensorF32 accepted a byte length not a multiple of 4")
	}
}

func TestFloat16ToFloat32_Good(t *testing.T) {
	cases := map[uint16]float32{
		0x0000: 0,
		0x3C00: 1,              // 1.0
		0xC000: -2,             // -2.0
		0x3555: 0.333251953125, // ~1/3
	}
	for bits, want := range cases {
		got := float16ToFloat32(bits)
		if got != want {
			t.Fatalf("float16ToFloat32(0x%04x) = %v, want %v", bits, got, want)
		}
	}
}

// TestFloat16ToFloat32_Bad proves +Inf's bit pattern widens to float32 +Inf (the exponent-all-ones,
// fraction-zero special case), not NaN or a finite value.
func TestFloat16ToFloat32_Bad(t *testing.T) {
	got := float16ToFloat32(0x7C00) // +Inf
	if !math.IsInf(float64(got), 1) {
		t.Fatalf("float16ToFloat32(+Inf bits) = %v, want +Inf", got)
	}
}

// TestFloat16ToFloat32_Ugly proves a subnormal f16 value (exponent zero, non-zero fraction) widens to
// the correct small normal float32 rather than zero or a garbage bit pattern.
func TestFloat16ToFloat32_Ugly(t *testing.T) {
	got := float16ToFloat32(0x0001) // smallest positive subnormal, 2^-24
	want := float32(1.0 / 16777216.0)
	if math.Abs(float64(got-want)) > 1e-12 {
		t.Fatalf("float16ToFloat32(smallest subnormal) = %v, want %v", got, want)
	}
}
