// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import (
	"math"
	"testing"

	core "dappco.re/go"
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

// addLoraTensors writes prefix+"lora.{0,2}.{weight,bias}" for a [in]->[low]->[out] LoRA-MLP.
func addLoraTensors(ts map[string]safetensors.Tensor, prefix string, in, low, out, seed int, bias bool) {
	ts[prefix+"lora.0.weight"] = bf16Tensor(syn(low*in, seed), low, in)
	ts[prefix+"lora.2.weight"] = bf16Tensor(syn(out*low, seed+1), out, low)
	if bias {
		ts[prefix+"lora.2.bias"] = bf16Tensor(syn(out, seed+2), out)
	}
}

// addLayerTensors populates one real-named RWKV-7 layer (li) into ts, matching the checkpoint layout
// LoadRWKV7Model expects: attn.{r,k,v,o}_proj, x_r..x_g, k_k/k_a/r_k, g_norm, the four LoRA gates
// (v_lora only when li>0), attn_norm/ffn_norm (+pre_norm at li==0), and ffn.{x_k,key,value}.
func addLayerTensors(ts map[string]safetensors.Tensor, li int, H, K, V, D, FF, decayLow, gateLow, aLow, vLow, seed int) {
	Dv := H * V
	p := core.Sprintf("model.layers.%d.", li)
	ts[p+"attn.r_proj.weight"] = bf16Tensor(syn(D*D, seed+1), D, D)
	ts[p+"attn.k_proj.weight"] = bf16Tensor(syn(D*D, seed+2), D, D)
	ts[p+"attn.v_proj.weight"] = bf16Tensor(syn(Dv*D, seed+3), Dv, D)
	ts[p+"attn.o_proj.weight"] = bf16Tensor(syn(D*Dv, seed+4), D, Dv)
	ts[p+"attn.x_r"] = bf16Tensor(syn(D, seed+5), 1, 1, D)
	ts[p+"attn.x_w"] = bf16Tensor(syn(D, seed+6), 1, 1, D)
	ts[p+"attn.x_k"] = bf16Tensor(syn(D, seed+7), 1, 1, D)
	ts[p+"attn.x_v"] = bf16Tensor(syn(D, seed+8), 1, 1, D)
	ts[p+"attn.x_a"] = bf16Tensor(syn(D, seed+9), 1, 1, D)
	ts[p+"attn.x_g"] = bf16Tensor(syn(D, seed+10), 1, 1, D)
	ts[p+"attn.k_k"] = bf16Tensor(syn(D, seed+11), D)
	ts[p+"attn.k_a"] = bf16Tensor(syn(D, seed+12), D)
	ts[p+"attn.r_k"] = bf16Tensor(syn(H*K, seed+13), H, K)
	ts[p+"attn.g_norm.weight"] = bf16Tensor(syn(Dv, seed+14), Dv)
	ts[p+"attn.g_norm.bias"] = bf16Tensor(syn(Dv, seed+15), Dv)
	addLoraTensors(ts, p+"attn.w_lora.", D, decayLow, D, seed+100, true)
	addLoraTensors(ts, p+"attn.a_lora.", D, aLow, D, seed+200, true)
	addLoraTensors(ts, p+"attn.g_lora.", D, gateLow, Dv, seed+300, false)
	if li > 0 {
		addLoraTensors(ts, p+"attn.v_lora.", D, vLow, Dv, seed+400, true)
	}
	if li == 0 {
		ts[p+"pre_norm.weight"] = bf16Tensor(syn(D, seed+16), D)
		ts[p+"pre_norm.bias"] = bf16Tensor(syn(D, seed+17), D)
	}
	ts[p+"attn_norm.weight"] = bf16Tensor(syn(D, seed+18), D)
	ts[p+"attn_norm.bias"] = bf16Tensor(syn(D, seed+19), D)
	ts[p+"ffn_norm.weight"] = bf16Tensor(syn(D, seed+20), D)
	ts[p+"ffn_norm.bias"] = bf16Tensor(syn(D, seed+21), D)
	ts[p+"ffn.x_k"] = bf16Tensor(syn(D, seed+22), D)
	ts[p+"ffn.key.weight"] = bf16Tensor(syn(FF*D, seed+23), FF, D)
	ts[p+"ffn.value.weight"] = bf16Tensor(syn(D*FF, seed+24), D, FF)
}

// mkCheckpoint builds a synthetic nLayers-deep RWKV-7 checkpoint's tensor map with real HF names/shapes.
func mkCheckpoint(H, K, V, D, FF, decayLow, gateLow, aLow, vLow, vocab, nLayers int) map[string]safetensors.Tensor {
	ts := map[string]safetensors.Tensor{
		"model.embeddings.weight": bf16Tensor(syn(vocab*D, 1), vocab, D),
		"model.norm.weight":       bf16Tensor(syn(D, 2), D),
		"model.norm.bias":         bf16Tensor(syn(D, 3), D),
	}
	for li := range nLayers {
		addLayerTensors(ts, li, H, K, V, D, FF, decayLow, gateLow, aLow, vLow, li*1000+10)
	}
	return ts
}

// TestLoader_LoadRWKV7Model_Good loads a synthetic 2-layer checkpoint (real HF tensor names/shapes),
// verifies the geometry is derived from the weight shapes (not guessed), and proves the loaded model
// decodes end-to-end.
func TestLoader_LoadRWKV7Model_Good(t *testing.T) {
	const H, K, V, D, FF = 2, 4, 3, 8, 16
	const decayLow, gateLow, aLow, vLow = 2, 2, 2, 2
	const vocab, nLayers = 32, 2
	ts := mkCheckpoint(H, K, V, D, FF, decayLow, gateLow, aLow, vLow, vocab, nLayers)

	m, err := LoadRWKV7Model(ts, []byte(`{"norm_eps": 1e-5}`))
	if err != nil {
		t.Fatalf("LoadRWKV7Model: %v", err)
	}
	want := BlockConfig{NumHeads: H, KeyDim: K, ValueDim: V}
	if m.Cfg != want {
		t.Fatalf("derived geometry %+v, want %+v", m.Cfg, want)
	}
	if m.D != D || m.Vocab != vocab || m.FF != FF || len(m.Layers) != nLayers {
		t.Fatalf("model dims wrong: D=%d vocab=%d FF=%d layers=%d", m.D, m.Vocab, m.FF, len(m.Layers))
	}
	if m.LMHead != nil {
		t.Error("LMHead should be nil (tied) — no lm_head.weight in the checkpoint")
	}
	if m.Layers[0].PreNormW == nil {
		t.Error("layer 0 must have pre_norm")
	}
	if m.Layers[1].PreNormW != nil {
		t.Error("layer 1 must NOT have pre_norm")
	}
	if m.Layers[0].Attn.VLora != nil {
		t.Error("layer 0 must NOT have v_lora")
	}
	if m.Layers[1].Attn.VLora == nil {
		t.Error("layer 1 must have v_lora")
	}

	gen, err := NewSession(m).Generate([]int32{1, 2, 3}, 4, -1)
	if err != nil {
		t.Fatalf("Generate on loaded model: %v", err)
	}
	if len(gen) != 4 {
		t.Fatalf("generated %d tokens, want 4", len(gen))
	}
	t.Logf("loaded synthetic RWKV-7 checkpoint: geometry %+v from shapes, %d layers, decodes end-to-end -> %v", m.Cfg, len(m.Layers), gen)
}

func TestLoader_LoadRWKV7Model_Bad(t *testing.T) {
	if _, err := LoadRWKV7Model(map[string]safetensors.Tensor{}, []byte(`{}`)); err == nil {
		t.Fatal("empty checkpoint accepted")
	}
}

// TestLoader_LoadRWKV7Model_Ugly proves the per-layer-uniform-geometry assumption is enforced: layer 1
// declaring a different head count than layer 0 must be rejected, not silently accepted with layer 0's
// geometry — distinct from _Bad's totally-empty checkpoint.
func TestLoader_LoadRWKV7Model_Ugly(t *testing.T) {
	const D, FF = 8, 16
	const decayLow, gateLow, aLow, vLow = 2, 2, 2, 2
	ts := map[string]safetensors.Tensor{
		"model.embeddings.weight": bf16Tensor(syn(32*D, 1), 32, D),
		"model.norm.weight":       bf16Tensor(syn(D, 2), D),
	}
	addLayerTensors(ts, 0, 2, 4, 3, D, FF, decayLow, gateLow, aLow, vLow, 10)   // H=2
	addLayerTensors(ts, 1, 4, 4, 3, D, FF, decayLow, gateLow, aLow, vLow, 2000) // H=4 — mismatch

	if _, err := LoadRWKV7Model(ts, []byte(`{}`)); err == nil {
		t.Fatal("layer 1 geometry differing from layer 0 accepted")
	}
}

func TestLoader_tensorF32_Good(t *testing.T) {
	got, err := tensorF32(bf16Tensor([]float32{1, -2, 0.5}, 3))
	if err != nil {
		t.Fatalf("tensorF32: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("len %d, want 3", len(got))
	}
}

func TestLoader_tensorF32_Bad(t *testing.T) {
	if _, err := tensorF32(safetensors.Tensor{Dtype: "I8", Data: []byte{1}}); err == nil {
		t.Fatal("unsupported dtype accepted")
	}
}

// TestLoader_tensorF32_Ugly rejects a malformed (odd) bf16 byte length rather than silently truncating.
func TestLoader_tensorF32_Ugly(t *testing.T) {
	if _, err := tensorF32(safetensors.Tensor{Dtype: "BF16", Data: []byte{1, 2, 3}}); err == nil {
		t.Fatal("odd bf16 byte length accepted")
	}
}

func TestLoader_loadLora_Good(t *testing.T) {
	ts := map[string]safetensors.Tensor{}
	addLoraTensors(ts, "x.", 8, 2, 8, 1, true)
	lo, err := loadLora(ts, "x.", true)
	if err != nil {
		t.Fatalf("loadLora: %v", err)
	}
	if lo.In != 8 || lo.Low != 2 || lo.Out != 8 || len(lo.Bias) != 8 {
		t.Fatalf("loaded lora shape wrong: %+v (bias len %d)", lo, len(lo.Bias))
	}
}

func TestLoader_loadLora_Bad(t *testing.T) {
	if _, err := loadLora(map[string]safetensors.Tensor{}, "x.", true); err == nil {
		t.Fatal("missing lora tensors accepted")
	}
}

// TestLoader_loadLora_Ugly proves wantBias=false (g_lora's shape) does not require lora.2.bias to be
// present, distinct from _Bad's fully-missing case.
func TestLoader_loadLora_Ugly(t *testing.T) {
	ts := map[string]safetensors.Tensor{}
	addLoraTensors(ts, "g.", 8, 2, 8, 1, false) // no bias written
	lo, err := loadLora(ts, "g.", false)
	if err != nil {
		t.Fatalf("loadLora (no-bias LoRA): %v", err)
	}
	if lo.Bias != nil {
		t.Fatalf("Bias = %v, want nil (wantBias=false)", lo.Bias)
	}
}
