// SPDX-Licence-Identifier: EUPL-1.2

package dbrx

import (
	"bytes"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// TestFactoryWeightNames_Good pins the DBRX factory tensor layout: the llama/mistral 2-norm coupling
// (post_attention_layernorm is the pre-MoE norm, no gemma-style sandwich norm), the router name
// (NormalizeWeights' alias target — see register.go), the packed-expert role names, and that the standard
// attention names are kept from StandardWeightNames (NormalizeWeights aliases DBRX's fused Wqkv and
// norm_attn_norm.* tensors onto exactly these names).
func TestFactoryWeightNames_Good(t *testing.T) {
	w := FactoryWeightNames()

	if w.MLPNorm != ".post_attention_layernorm.weight" {
		t.Errorf("MLPNorm = %q, want .post_attention_layernorm.weight (llama/mistral 2-norm layout)", w.MLPNorm)
	}
	if w.PostAttnNorm != "" {
		t.Errorf("PostAttnNorm = %q, want \"\" — a non-empty value would apply a second (gemma-style) norm DBRX does not carry", w.PostAttnNorm)
	}
	if w.NormBiasOne {
		t.Error("NormBiasOne = true, want false (DBRX is plain RMSNorm, not gemma's +1 fold)")
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

	if w.AttnNorm != ".input_layernorm.weight" {
		t.Errorf("AttnNorm = %q, want .input_layernorm.weight", w.AttnNorm)
	}
	if w.Q != ".self_attn.q_proj" || w.K != ".self_attn.k_proj" || w.V != ".self_attn.v_proj" || w.O != ".self_attn.o_proj" {
		t.Errorf("attention projections = %q/%q/%q/%q, want .self_attn.{q,k,v,o}_proj", w.Q, w.K, w.V, w.O)
	}
}

// TestFactoryWeightNames_Bad guards against DBRX acquiring a shared expert by accident — real DBRX
// checkpoints carry no shared_expert.* tensors, so a stray non-empty name here would make assembleMoE probe
// a tensor that never exists (harmless — LoadLinear is nil-safe on a missing name — but a silent
// declaration drift worth pinning).
func TestFactoryWeightNames_Bad(t *testing.T) {
	w := FactoryWeightNames()
	if w.MoE.SharedGate != "" || w.MoE.SharedUp != "" || w.MoE.SharedDown != "" || w.MoE.SharedSigmoid != "" {
		t.Fatalf("DBRX declared a shared expert it does not have: %+v", w.MoE)
	}
}

// TestFactoryWeightNames_Ugly proves the packed-expert names never collide with the router or with each
// other — a collision would silently alias two roles onto one tensor.
func TestFactoryWeightNames_Ugly(t *testing.T) {
	w := FactoryWeightNames()
	names := map[string]string{"router": w.MoE.Router, "gate": w.MoE.ExpGate, "up": w.MoE.ExpUp, "down": w.MoE.ExpDown}
	seen := make(map[string]string, len(names))
	for role, name := range names {
		if other, ok := seen[name]; ok {
			t.Fatalf("role %q and %q share the same tensor name %q", role, other, name)
		}
		seen[name] = role
	}
}

// bf16Tensor2D builds a deterministic 2-D bf16 tensor whose bytes encode (row, col) so packExperts' output
// can be checked byte-for-byte against a hand-computed expectation, not just a shape/length probe.
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

// normalisedExpertCheckpoint builds a minimal 2-layer, numExperts-expert tensor set already in
// NormalizeWeights' per-expert output convention (model.layers.{l}.mlp.experts.{e}.{gate,up,down}_proj.weight
// — the shape packExperts consumes; NormalizeWeights' own decode of DBRX's fused w1/v1/w2 layout is tested
// separately in register_test.go). Every expert's tensor is distinct and deterministically seeded (seed =
// layer*100 + expert*10 + role) so packExperts' concatenation ORDER is provable, not just its total byte
// count.
func normalisedExpertCheckpoint(numLayers, numExperts, ff, hidden int) map[string]safetensors.Tensor {
	out := make(map[string]safetensors.Tensor)
	for layer := 0; layer < numLayers; layer++ {
		for e := 0; e < numExperts; e++ {
			base := uint16(layer*100 + e*10)
			prefix := core.Sprintf("model.layers.%d.mlp.experts.%d.", layer, e)
			out[prefix+"gate_proj.weight"] = bf16Tensor2D(ff, hidden, base+1)
			out[prefix+"up_proj.weight"] = bf16Tensor2D(ff, hidden, base+2)
			out[prefix+"down_proj.weight"] = bf16Tensor2D(hidden, ff, base+3)
		}
	}
	return out
}

// TestPackExperts_Good proves packExperts' packed tensor is BYTE-IDENTICAL to the row-major concatenation
// of the source per-expert tensors, in expert-index order, for every layer and every role — the "same
// tensor maps" half of the #18 parity method: the packed bytes ARE the checkpoint's (already-normalised)
// bytes, just relocated under one name, never altered.
func TestPackExperts_Good(t *testing.T) {
	const numLayers, numExperts, ff, hidden = 2, 3, 8, 4
	in := normalisedExpertCheckpoint(numLayers, numExperts, ff, hidden)
	origLen := len(in)

	out, err := packExperts(in, numLayers, numExperts)
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

	for layer := 0; layer < numLayers; layer++ {
		for _, role := range expertRoles {
			var want []byte
			var outDim, inDim int
			for e := 0; e < numExperts; e++ {
				src := in[core.Sprintf("model.layers.%d.mlp.experts.%d.%s.weight", layer, e, role)]
				want = append(want, src.Data...)
				outDim = src.Shape[0]
				inDim = src.Shape[1]
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
}

// TestPackExperts_QuantisedScalesAndBiases_Good proves packExperts also packs a quantised role's
// .scales/.biases sibling tensors, in the SAME row-major expert-index order as .weight — the
// packed-scales/biases convention model.Assemble's quant loader (LoadLinear/affineGeometry) reads
// off a natively-packed checkpoint (gemma4's switch_glu.*.weight/.scales/.biases), synthesised here
// from DBRX's (already-normalised) per-expert quant triples so a quantised DBRX checkpoint can serve
// too (#59).
func TestPackExperts_QuantisedScalesAndBiases_Good(t *testing.T) {
	const numLayers, numExperts, ff, hidden, groups = 1, 3, 8, 4, 2
	in := normalisedExpertCheckpoint(numLayers, numExperts, ff, hidden)
	for e := 0; e < numExperts; e++ {
		base := uint16(e * 10)
		prefix := core.Sprintf("model.layers.0.mlp.experts.%d.", e)
		in[prefix+"gate_proj.scales"], in[prefix+"gate_proj.biases"] = bf16Tensor2D(ff, groups, base+201), bf16Tensor2D(ff, groups, base+202)
		in[prefix+"up_proj.scales"], in[prefix+"up_proj.biases"] = bf16Tensor2D(ff, groups, base+203), bf16Tensor2D(ff, groups, base+204)
		in[prefix+"down_proj.scales"], in[prefix+"down_proj.biases"] = bf16Tensor2D(hidden, groups, base+205), bf16Tensor2D(hidden, groups, base+206)
	}

	out, err := packExperts(in, numLayers, numExperts)
	if err != nil {
		t.Fatalf("packExperts: %v", err)
	}
	for _, role := range expertRoles {
		var wantScales, wantBiases []byte
		var outDim int
		for e := 0; e < numExperts; e++ {
			sSrc := in[core.Sprintf("model.layers.0.mlp.experts.%d.%s.scales", e, role)]
			bSrc := in[core.Sprintf("model.layers.0.mlp.experts.%d.%s.biases", e, role)]
			wantScales = append(wantScales, sSrc.Data...)
			wantBiases = append(wantBiases, bSrc.Data...)
			outDim = sSrc.Shape[0]
		}
		packedName := core.Sprintf("model.layers.0%s.%s", packedExpertsPrefix, role)
		scales, ok := out[packedName+".scales"]
		if !ok || !bytes.Equal(scales.Data, wantScales) {
			t.Fatalf("role %s: packed scales missing or != concatenation of experts 0..%d in order", role, numExperts-1)
		}
		if len(scales.Shape) != 2 || scales.Shape[0] != outDim*numExperts || scales.Shape[1] != groups {
			t.Fatalf("role %s: packed scales shape = %v, want [%d,%d]", role, scales.Shape, outDim*numExperts, groups)
		}
		biases, ok := out[packedName+".biases"]
		if !ok || !bytes.Equal(biases.Data, wantBiases) {
			t.Fatalf("role %s: packed biases missing or != concatenation of experts 0..%d in order", role, numExperts-1)
		}
	}
}

// TestPackExperts_QuantisedScalesAndBiases_Bad proves a partially-quantised expert (scales present,
// biases absent) fails loudly rather than silently packing a scales-only triple, and that a
// mismatched scales shape across experts is caught the same way the weight shape check already is.
func TestPackExperts_QuantisedScalesAndBiases_Bad(t *testing.T) {
	t.Run("biases_absent", func(t *testing.T) {
		in := normalisedExpertCheckpoint(1, 2, 8, 4)
		in["model.layers.0.mlp.experts.0.gate_proj.scales"] = bf16Tensor2D(8, 2, 1)
		if _, err := packExperts(in, 1, 2); err == nil {
			t.Fatal("packExperts accepted a role with scales but no biases")
		}
	})
	t.Run("scales_shape_mismatch", func(t *testing.T) {
		in := normalisedExpertCheckpoint(1, 2, 8, 4)
		for _, e := range []int{0, 1} {
			prefix := core.Sprintf("model.layers.0.mlp.experts.%d.", e)
			in[prefix+"gate_proj.scales"] = bf16Tensor2D(8, 2, uint16(e))
			in[prefix+"gate_proj.biases"] = bf16Tensor2D(8, 2, uint16(e))
		}
		in["model.layers.0.mlp.experts.1.gate_proj.scales"] = bf16Tensor2D(8, 3, 9) // wrong group count
		if _, err := packExperts(in, 1, 2); err == nil {
			t.Fatal("packExperts accepted experts whose scales shapes disagree")
		}
	})
}

// TestPackExperts_Bad proves a genuinely malformed/partial checkpoint fails LOUDLY, naming the exact tensor
// that broke the pattern, rather than silently packing a short or misaligned tensor.
func TestPackExperts_Bad(t *testing.T) {
	t.Run("missing_expert", func(t *testing.T) {
		in := normalisedExpertCheckpoint(1, 3, 8, 4)
		delete(in, "model.layers.0.mlp.experts.2.gate_proj.weight")
		if _, err := packExperts(in, 1, 3); err == nil {
			t.Fatal("packExperts accepted a checkpoint missing expert 2's gate weight")
		} else if !core.Contains(err.Error(), "experts.2.gate_proj.weight") {
			t.Errorf("error %q does not name the missing tensor", err.Error())
		}
	})
	t.Run("shape_mismatch", func(t *testing.T) {
		in := normalisedExpertCheckpoint(1, 2, 8, 4)
		in["model.layers.0.mlp.experts.1.gate_proj.weight"] = bf16Tensor2D(8, 5, 0) // wrong inDim
		if _, err := packExperts(in, 1, 2); err == nil {
			t.Fatal("packExperts accepted an expert with a mismatched shape")
		}
	})
	t.Run("dtype_mismatch", func(t *testing.T) {
		in := normalisedExpertCheckpoint(1, 2, 8, 4)
		bad := in["model.layers.0.mlp.experts.1.gate_proj.weight"]
		bad.Dtype = "F32"
		in["model.layers.0.mlp.experts.1.gate_proj.weight"] = bad
		if _, err := packExperts(in, 1, 2); err == nil {
			t.Fatal("packExperts accepted experts with mismatched dtypes")
		}
	})
	t.Run("no_experts_at_all", func(t *testing.T) {
		if _, err := packExperts(map[string]safetensors.Tensor{}, 1, 2); err == nil {
			t.Fatal("packExperts accepted a checkpoint with no expert tensors")
		}
	})
}

// TestPackExperts_Ugly covers the degenerate inputs: a non-positive layer or expert count must be rejected
// before any lookup (never a nonsensical zero-sized allocation or an infinite loop), and an empty-but-non-nil
// input map degrades gracefully.
func TestPackExperts_Ugly(t *testing.T) {
	for _, tc := range []struct {
		name              string
		numLayers, numExp int
	}{
		{"zero_layers", 0, 8}, {"zero_experts", 2, 0}, {"negative_layers", -1, 8}, {"negative_experts", 2, -3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := packExperts(normalisedExpertCheckpoint(2, 8, 4, 4), tc.numLayers, tc.numExp); err == nil {
				t.Fatalf("packExperts(numLayers=%d, numExperts=%d) accepted", tc.numLayers, tc.numExp)
			}
		})
	}
	if _, err := packExperts(nil, 1, 1); err == nil {
		t.Fatal("packExperts accepted a nil tensor map")
	}
}
