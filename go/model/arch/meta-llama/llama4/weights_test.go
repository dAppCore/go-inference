// SPDX-Licence-Identifier: EUPL-1.2

package llama4

import (
	"bytes"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// TestFactoryWeightNames_Good pins the Llama 4 factory tensor layout: the llama 2-norm coupling
// (post_attention_layernorm is the pre-MoE norm, no gemma-style sandwich norm), the router and
// packed-expert role names (NormalizeWeights' alias targets — see register.go), the shared-expert
// trio (llama4, unlike dbrx/mixtral/granitemoe, always carries one), and that the standard attention
// names are kept from StandardWeightNames (NormalizeWeights never touches self_attn.* or
// input_layernorm.weight — the real checkpoint already ships those llama-shaped).
func TestFactoryWeightNames_Good(t *testing.T) {
	w := FactoryWeightNames()

	if w.MLPNorm != ".post_attention_layernorm.weight" {
		t.Errorf("MLPNorm = %q, want .post_attention_layernorm.weight (llama 2-norm layout)", w.MLPNorm)
	}
	if w.PostAttnNorm != "" {
		t.Errorf("PostAttnNorm = %q, want \"\" — a non-empty value would apply a second (gemma-style) norm Llama 4 does not carry", w.PostAttnNorm)
	}
	if w.NormBiasOne {
		t.Error("NormBiasOne = true, want false (Llama 4 is plain RMSNorm, not gemma's +1 fold)")
	}

	if w.MoE.Router != ".mlp.gate" {
		t.Errorf("MoE.Router = %q, want .mlp.gate", w.MoE.Router)
	}
	if w.MoE.ExpGate != packedExpertsPrefix+".gate_proj" || w.MoE.ExpUp != packedExpertsPrefix+".up_proj" || w.MoE.ExpDown != packedExpertsPrefix+".down_proj" {
		t.Errorf("MoE routed experts = %q/%q/%q, want packExperts' synthesised %s.{gate,up,down}_proj", w.MoE.ExpGate, w.MoE.ExpUp, w.MoE.ExpDown, packedExpertsPrefix)
	}
	if w.MoE.PreFFNorm != ".post_attention_layernorm.weight" {
		t.Errorf("MoE.PreFFNorm = %q, want .post_attention_layernorm.weight", w.MoE.PreFFNorm)
	}
	if w.MoE.SharedGate != ".mlp.shared_expert.gate_proj" || w.MoE.SharedUp != ".mlp.shared_expert.up_proj" || w.MoE.SharedDown != ".mlp.shared_expert.down_proj" {
		t.Errorf("MoE shared expert = %q/%q/%q, want .mlp.shared_expert.{gate,up,down}_proj", w.MoE.SharedGate, w.MoE.SharedUp, w.MoE.SharedDown)
	}

	if w.AttnNorm != ".input_layernorm.weight" {
		t.Errorf("AttnNorm = %q, want .input_layernorm.weight", w.AttnNorm)
	}
	if w.Q != ".self_attn.q_proj" || w.K != ".self_attn.k_proj" || w.V != ".self_attn.v_proj" || w.O != ".self_attn.o_proj" {
		t.Errorf("attention projections = %q/%q/%q/%q, want .self_attn.{q,k,v,o}_proj", w.Q, w.K, w.V, w.O)
	}
}

// TestFactoryWeightNames_Bad guards against Llama 4's shared expert acquiring a sigmoid gate it does
// not have: unlike Qwen1.5-MoE (mlp.shared_expert_gate.weight), the real Llama4-Scout index excerpt
// (testdata/) carries only the shared_expert.{gate,up,down}_proj trio — no gate tensor — so
// composed/moe.go's documented ungated fallback is what a real checkpoint exercises. A stray
// non-empty SharedSigmoid here would make assembleMoE probe a tensor that never exists (harmless —
// LoadLinear is nil-safe on a missing name — but a silent declaration drift worth pinning).
func TestFactoryWeightNames_Bad(t *testing.T) {
	w := FactoryWeightNames()
	if w.MoE.SharedSigmoid != "" {
		t.Fatalf("SharedSigmoid = %q, want \"\" — Llama 4's shared expert has no sigmoid gate tensor", w.MoE.SharedSigmoid)
	}
}

// TestFactoryWeightNames_Ugly proves the packed-expert and shared-expert names never collide with
// the router or with each other — a collision would silently alias two roles onto one tensor.
func TestFactoryWeightNames_Ugly(t *testing.T) {
	w := FactoryWeightNames()
	names := map[string]string{
		"router": w.MoE.Router, "gate": w.MoE.ExpGate, "up": w.MoE.ExpUp, "down": w.MoE.ExpDown,
		"shared_gate": w.MoE.SharedGate, "shared_up": w.MoE.SharedUp, "shared_down": w.MoE.SharedDown,
	}
	seen := make(map[string]string, len(names))
	for role, name := range names {
		if other, ok := seen[name]; ok {
			t.Fatalf("role %q and %q share the same tensor name %q", role, other, name)
		}
		seen[name] = role
	}
}

// bf16Tensor2D builds a deterministic 2-D bf16 tensor whose bytes encode (row, col, seed) so
// packExperts' output can be checked byte-for-byte against a hand-computed expectation, not just a
// shape/length probe.
func bf16Tensor2D(rows, cols int, seed uint16) safetensors.Tensor {
	data := make([]byte, rows*cols*2)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			v := seed + uint16(r*cols+c)
			i := (r*cols + c) * 2
			data[i], data[i+1] = byte(v), byte(v>>8)
		}
	}
	return safetensors.Tensor{Dtype: "BF16", Shape: []int{rows, cols}, Data: data}
}

// llama4NormalisedCheckpoint builds a minimal multi-layer tensor set already in NormalizeWeights'
// per-expert output convention (language_model.model.layers.{l}.mlp.experts.{e}.{gate,up,down}_proj.
// weight — the shape packExperts consumes; NormalizeWeights' own decode of the real packed bmm
// layout is tested separately in register_test.go). moeLayers selects which layers carry routed
// experts at all — the rest are left dense, Llama 4's interleaved moe_layers wrinkle none of
// dbrx/qwenmoe/mixtral's all-MoE checkpoints have. Every expert's tensor is distinct and
// deterministically seeded (seed = layer*100 + expert*10 + role) so packExperts' concatenation ORDER
// is provable, not just its total byte count. Keys carry the real checkpoint's "language_model."
// wrapper prefix (see FactoryWeightNames' doc) — packExperts must resolve it itself.
func llama4NormalisedCheckpoint(numLayers, numExperts, ff, hidden int, moeLayers map[int]bool) map[string]safetensors.Tensor {
	out := make(map[string]safetensors.Tensor)
	for layer := 0; layer < numLayers; layer++ {
		if !moeLayers[layer] {
			continue
		}
		for e := 0; e < numExperts; e++ {
			base := uint16(layer*100 + e*10)
			prefix := core.Sprintf("language_model.model.layers.%d.mlp.experts.%d.", layer, e)
			out[prefix+"gate_proj.weight"] = bf16Tensor2D(ff, hidden, base+1)
			out[prefix+"up_proj.weight"] = bf16Tensor2D(ff, hidden, base+2)
			out[prefix+"down_proj.weight"] = bf16Tensor2D(hidden, ff, base+3)
		}
	}
	return out
}

// archWithMoELayers builds a minimal model.Arch carrying numLayers layers, numExperts routed
// experts, with only the layer indices in moe marked MoE — everything else left dense.
func archWithMoELayers(numLayers, numExperts int, moe ...int) model.Arch {
	moeSet := make(map[int]bool, len(moe))
	for _, l := range moe {
		moeSet[l] = true
	}
	layers := make([]model.LayerSpec, numLayers)
	for i := range layers {
		layers[i].MoE = moeSet[i]
	}
	return model.Arch{Experts: numExperts, Layer: layers}
}

// TestPackExperts_Good proves packExperts' packed tensor is BYTE-IDENTICAL to the row-major
// concatenation of the source per-expert tensors, in expert-index order, for every MoE layer — the
// "same tensor maps" half of the #18 parity method: the packed bytes ARE the checkpoint's
// (already-normalised) bytes, just relocated under one name, never altered — AND that dense layers
// (moe_layers interleaving) are skipped, not packed.
func TestPackExperts_Good(t *testing.T) {
	const numLayers, numExperts, ff, hidden = 3, 3, 8, 4
	moe := map[int]bool{0: true, 2: true} // layer 1 is dense
	arch := archWithMoELayers(numLayers, numExperts, 0, 2)
	in := llama4NormalisedCheckpoint(numLayers, numExperts, ff, hidden, moe)
	origLen := len(in)

	out, err := packExperts(in, arch)
	if err != nil {
		t.Fatalf("packExperts: %v", err)
	}
	if len(in) != origLen {
		t.Fatalf("packExperts mutated its input map: len = %d, want %d", len(in), origLen)
	}
	for name, tensor := range in {
		got, ok := out[name]
		if !ok || !bytes.Equal(got.Data, tensor.Data) {
			t.Fatalf("source tensor %q missing or altered in packExperts output", name)
		}
	}

	for layer := range moe {
		for _, role := range expertRoles {
			var want []byte
			var outDim, inDim int
			for e := 0; e < numExperts; e++ {
				src := in[core.Sprintf("language_model.model.layers.%d.mlp.experts.%d.%s.weight", layer, e, role)]
				want = append(want, src.Data...)
				outDim, inDim = src.Shape[0], src.Shape[1]
			}
			packedName := core.Sprintf("model.layers.%d%s.%s.weight", layer, packedExpertsPrefix, role)
			packed, ok := out[packedName]
			if !ok {
				t.Fatalf("layer %d role %s: packed tensor %q absent", layer, role, packedName)
			}
			if !bytes.Equal(packed.Data, want) {
				t.Fatalf("layer %d role %s: packed bytes != concatenation of experts 0..%d in order", layer, role, numExperts-1)
			}
			if len(packed.Shape) != 2 || packed.Shape[0] != outDim*numExperts || packed.Shape[1] != inDim {
				t.Fatalf("layer %d role %s: packed shape = %v, want [%d,%d]", layer, role, packed.Shape, outDim*numExperts, inDim)
			}
			if packed.Dtype != "BF16" {
				t.Fatalf("layer %d role %s: packed dtype = %q, want BF16", layer, role, packed.Dtype)
			}
		}
	}

	// Layer 1 is dense (not in moe_layers) — no packed tensor should exist for it.
	for _, role := range expertRoles {
		name := core.Sprintf("model.layers.1%s.%s.weight", packedExpertsPrefix, role)
		if _, ok := out[name]; ok {
			t.Fatalf("layer 1 is dense but packExperts emitted %q", name)
		}
	}
}

// TestPackExperts_Bad proves a genuinely malformed/partial checkpoint (on a layer DECLARED MoE)
// fails LOUDLY, naming the exact tensor that broke the pattern, rather than silently packing a short
// or misaligned tensor.
func TestPackExperts_Bad(t *testing.T) {
	t.Run("missing_expert", func(t *testing.T) {
		in := llama4NormalisedCheckpoint(1, 3, 8, 4, map[int]bool{0: true})
		delete(in, "language_model.model.layers.0.mlp.experts.2.gate_proj.weight")
		if _, err := packExperts(in, archWithMoELayers(1, 3, 0)); err == nil {
			t.Fatal("packExperts accepted a checkpoint missing expert 2's gate weight")
		} else if !core.Contains(err.Error(), "experts.2.gate_proj.weight") {
			t.Errorf("error %q does not name the missing tensor", err.Error())
		}
	})
	t.Run("shape_mismatch", func(t *testing.T) {
		in := llama4NormalisedCheckpoint(1, 2, 8, 4, map[int]bool{0: true})
		in["language_model.model.layers.0.mlp.experts.1.gate_proj.weight"] = bf16Tensor2D(8, 5, 0) // wrong inDim
		if _, err := packExperts(in, archWithMoELayers(1, 2, 0)); err == nil {
			t.Fatal("packExperts accepted an expert with a mismatched shape")
		}
	})
	t.Run("dtype_mismatch", func(t *testing.T) {
		in := llama4NormalisedCheckpoint(1, 2, 8, 4, map[int]bool{0: true})
		bad := in["language_model.model.layers.0.mlp.experts.1.gate_proj.weight"]
		bad.Dtype = "F32"
		in["language_model.model.layers.0.mlp.experts.1.gate_proj.weight"] = bad
		if _, err := packExperts(in, archWithMoELayers(1, 2, 0)); err == nil {
			t.Fatal("packExperts accepted experts with mismatched dtypes")
		}
	})
	t.Run("no_experts_on_a_declared_moe_layer", func(t *testing.T) {
		if _, err := packExperts(map[string]safetensors.Tensor{}, archWithMoELayers(1, 2, 0)); err == nil {
			t.Fatal("packExperts accepted a declared-MoE layer with no expert tensors at all")
		}
	})
	t.Run("zero_experts", func(t *testing.T) {
		if _, err := packExperts(map[string]safetensors.Tensor{}, model.Arch{Experts: 0, Layer: []model.LayerSpec{{MoE: true}}}); err == nil {
			t.Fatal("packExperts accepted arch.Experts == 0")
		}
	})
}

// TestPackExperts_Ugly covers surprising-but-valid inputs: an arch with NO MoE layers at all (every
// layer dense) must pass through cleanly — packing nothing is correct, not an error — and a
// checkpoint that never carried the "language_model." multimodal wrapper prefix (a standalone
// llama4_text-only pack) must still pack correctly, since model.NormalizeWrapperNames is a documented
// no-op when the prefix is absent.
func TestPackExperts_Ugly(t *testing.T) {
	t.Run("no_moe_layers", func(t *testing.T) {
		in := llama4NormalisedCheckpoint(2, 2, 8, 4, map[int]bool{}) // no layer carries experts
		out, err := packExperts(in, archWithMoELayers(2, 2))
		if err != nil {
			t.Fatalf("packExperts on an all-dense arch: %v", err)
		}
		if len(out) != len(in) {
			t.Fatalf("packExperts on an all-dense arch added %d tensor(s), want 0", len(out)-len(in))
		}
	})
	t.Run("unwrapped_checkpoint", func(t *testing.T) {
		const numLayers, numExperts, ff, hidden = 1, 2, 8, 4
		wrapped := llama4NormalisedCheckpoint(numLayers, numExperts, ff, hidden, map[int]bool{0: true})
		const pfx = "language_model."
		unwrapped := make(map[string]safetensors.Tensor, len(wrapped))
		for name, tensor := range wrapped {
			unwrapped[name[len(pfx):]] = tensor
		}
		out, err := packExperts(unwrapped, archWithMoELayers(numLayers, numExperts, 0))
		if err != nil {
			t.Fatalf("packExperts on an unwrapped checkpoint: %v", err)
		}
		if _, ok := out[core.Sprintf("model.layers.0%s.gate_proj.weight", packedExpertsPrefix)]; !ok {
			t.Fatal("packExperts did not pack an already-unprefixed checkpoint")
		}
	})
}
