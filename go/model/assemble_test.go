// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// TestAssemble_StandardWeightNames_Good spot-checks the canonical HF layout's model-level
// and per-layer suffix names — the superset every architecture with the plain HF naming
// uses as-is.
func TestAssemble_StandardWeightNames_Good(t *testing.T) {
	n := StandardWeightNames()
	if n.Embed != "model.embed_tokens" || n.LMHead != "lm_head" || n.FinalNorm != "model.norm.weight" {
		t.Fatalf("model-level names = %+v, want the canonical HF ids", n)
	}
	if n.LayerPrefix != "model.layers.%d" {
		t.Fatalf("LayerPrefix = %q, want %q", n.LayerPrefix, "model.layers.%d")
	}
	if n.Q != ".self_attn.q_proj" || n.O != ".self_attn.o_proj" {
		t.Fatalf("attention suffixes = Q:%q O:%q, want the self_attn.* suffixes", n.Q, n.O)
	}
}

// TestAssemble_StandardWeightNames_Bad covers the override pattern the type doc describes:
// an arch with different names copies the struct and overrides only the fields that
// differ, so every other Standard default survives untouched.
func TestAssemble_StandardWeightNames_Bad(t *testing.T) {
	custom := StandardWeightNames()
	custom.AttnNorm = ".custom_norm.weight"
	if custom.AttnNorm != ".custom_norm.weight" {
		t.Fatalf("override did not take: AttnNorm = %q", custom.AttnNorm)
	}
	if custom.Q != ".self_attn.q_proj" || custom.LayerPrefix != "model.layers.%d" {
		t.Fatalf("overriding one field must not disturb the others: %+v", custom)
	}
	// the original constructor call is unaffected by mutating the copy.
	if fresh := StandardWeightNames(); fresh.AttnNorm != ".input_layernorm.weight" {
		t.Fatalf("StandardWeightNames() = %+v, mutated by a prior copy's override", fresh)
	}
}

// TestAssemble_StandardWeightNames_Ugly covers the deepest nested block (MoE) — the one
// most likely left blank by an override that forgets it: every MoE suffix the router and
// dual-branch FFN need is populated, non-empty.
func TestAssemble_StandardWeightNames_Ugly(t *testing.T) {
	n := StandardWeightNames()
	fields := map[string]string{
		"RouterScale": n.MoE.RouterScale, "PerExpertScale": n.MoE.PerExpertScale,
		"Router": n.MoE.Router, "ExpGate": n.MoE.ExpGate, "ExpUp": n.MoE.ExpUp,
		"ExpGateUp": n.MoE.ExpGateUp, "ExpDown": n.MoE.ExpDown,
	}
	for name, v := range fields {
		if v == "" {
			t.Fatalf("MoE.%s is empty, want a populated suffix", name)
		}
	}
}

// minimalDenseTensors builds a hermetic single-layer, dense-MLP tensor set for Assemble:
// hidden=4, one layer, no MoE, no PLE tower — every weight ValidateRequired demands, plus
// FinalNorm, under simple custom weight names (not the HF Standard layout, to keep the
// fixture small and self-explanatory).
func minimalDenseTensors(dtype string) map[string]safetensors.Tensor {
	vec := func(n int) safetensors.Tensor {
		return safetensors.Tensor{Shape: []int{n}, Data: make([]byte, n*2), Dtype: dtype}
	}
	mat := func(rows, cols int) safetensors.Tensor {
		return safetensors.Tensor{Shape: []int{rows, cols}, Data: make([]byte, rows*cols*2)}
	}
	return map[string]safetensors.Tensor{
		"embed.weight":             mat(8, 4),
		"norm.weight":              vec(4),
		"layer.0.attn_norm.weight": vec(4),
		"layer.0.q.weight":         mat(4, 4),
		"layer.0.k.weight":         mat(4, 4),
		"layer.0.o.weight":         mat(4, 4),
		"layer.0.mlp_norm.weight":  vec(4),
		"layer.0.gate.weight":      mat(8, 4),
		"layer.0.up.weight":        mat(8, 4),
		"layer.0.down.weight":      mat(4, 8),
	}
}

func minimalDenseNames() WeightNames {
	return WeightNames{
		Embed: "embed", FinalNorm: "norm.weight",
		LayerPrefix: "layer.%d",
		AttnNorm:    ".attn_norm.weight",
		Q:           ".q", K: ".k", O: ".o",
		MLPNorm: ".mlp_norm.weight", Gate: ".gate", Up: ".up", Down: ".down",
	}
}

// FF is deliberately a value the gate.weight tensor's shape does NOT agree with (the gate
// resolves OutDim=8 from its shape) — TestAssemble_Assemble_Good pins that the per-layer
// FFN width used to load Down comes from the shape, not this field.
func minimalDenseArch() Arch {
	return Arch{
		Hidden: 4, Heads: 2, FF: 99,
		Layer: []LayerSpec{{Attention: GlobalAttention, CacheIndex: 0, KVShareFrom: 0, HeadDim: 2, KVHeads: 2}},
	}
}

// TestAssemble_Assemble_Good covers the ordinary single-layer dense-MLP build: every
// required weight resolves, the per-layer FFN width is read from the gate's actual OutDim
// (MatFormer-style), and the fold-on-load convention (NormBiasOne) applies cleanly over
// BF16 norm bytes.
func TestAssemble_Assemble_Good(t *testing.T) {
	names := minimalDenseNames()
	names.NormBiasOne = true
	m, err := Assemble(minimalDenseTensors("BF16"), minimalDenseArch(), names)
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if m.Embed == nil || m.FinalNorm == nil {
		t.Fatalf("Assemble result missing Embed/FinalNorm: %+v", m)
	}
	if len(m.Layers) != 1 {
		t.Fatalf("len(Layers) = %d, want 1", len(m.Layers))
	}
	L := m.Layers[0]
	if L.Q == nil || L.O == nil || L.K == nil {
		t.Fatalf("layer 0 missing a required attention weight: %+v", L)
	}
	if L.Gate == nil || L.Up == nil || L.Down == nil || L.MoE != nil {
		t.Fatalf("layer 0 should be dense (Gate/Up/Down set, MoE nil): %+v", L)
	}
	if L.Down.InDim != L.Gate.OutDim {
		t.Fatalf("Down.InDim = %d, want the gate's OutDim %d (per-layer FFN width read from the shape)", L.Down.InDim, L.Gate.OutDim)
	}
	if L.Down.InDim == 99 {
		t.Fatal("Down.InDim came from arch.FF (99), not the gate's actual shape — MatFormer per-layer width broken")
	}
	if m.Tied() != (m.LMHead == nil) {
		t.Fatalf("Tied() = %v inconsistent with LMHead=%v", m.Tied(), m.LMHead)
	}
}

// TestAssemble_Assemble_Bad covers the always-required Embed weight absent: a malformed
// checkpoint is a clean load error naming the missing tensor, never a nil-deref later.
func TestAssemble_Assemble_Bad(t *testing.T) {
	tensors := minimalDenseTensors("BF16")
	delete(tensors, "embed.weight")
	if _, err := Assemble(tensors, minimalDenseArch(), minimalDenseNames()); err == nil {
		t.Fatal("Assemble with no embed tensor: expected an error")
	}
}

// TestAssemble_Assemble_Ugly covers the norm-bias-fold failure path: NormBiasOne set on a
// norm tensor whose dtype foldNormBiasOne doesn't support surfaces as a clean Assemble
// error (not a panic on the unsupported bytes), even with zero decode layers.
func TestAssemble_Assemble_Ugly(t *testing.T) {
	names := WeightNames{Embed: "embed", FinalNorm: "norm.weight", NormBiasOne: true}
	tensors := map[string]safetensors.Tensor{
		"embed.weight": {Shape: []int{8, 4}, Data: make([]byte, 8*4*2)},
		"norm.weight":  {Shape: []int{4}, Data: make([]byte, 4*2), Dtype: "int8"}, // unsupported by foldNormBiasOne
	}
	if _, err := Assemble(tensors, Arch{Hidden: 4}, names); err == nil {
		t.Fatal("Assemble with an unsupported norm-bias dtype: expected an error")
	}
}

func TestAssemble_TieWordEmbeddings_Good(t *testing.T) {
	for _, tc := range []struct {
		name              string
		declaration       *bool
		includeHead, tied bool
	}{
		{name: "unspecified remains compatible", declaration: nil, tied: true},
		{name: "declared tied", declaration: boolPointer(true), tied: true},
		{name: "declared untied", declaration: boolPointer(false), includeHead: true, tied: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tensors := minimalDenseTensors("BF16")
			names := minimalDenseNames()
			names.LMHead = "head"
			if tc.includeHead {
				tensors["head.weight"] = safetensors.Tensor{Shape: []int{8, 4}, Data: make([]byte, 8*4*2)}
			}
			arch := minimalDenseArch()
			arch.TieWordEmbeddings = tc.declaration
			m, err := Assemble(tensors, arch, names)
			if err != nil {
				t.Fatalf("Assemble: %v", err)
			}
			if m.Tied() != tc.tied {
				t.Fatalf("Tied() = %v, want %v", m.Tied(), tc.tied)
			}
		})
	}
}

func TestAssemble_TieWordEmbeddings_Bad(t *testing.T) {
	for _, tc := range []struct {
		declaredTied, includeHead bool
	}{
		{declaredTied: true, includeHead: true},
		{declaredTied: false, includeHead: false},
	} {
		tensors := minimalDenseTensors("BF16")
		names := minimalDenseNames()
		names.LMHead = "head"
		if tc.includeHead {
			tensors["head.weight"] = safetensors.Tensor{Shape: []int{8, 4}, Data: make([]byte, 8*4*2)}
		}
		arch := minimalDenseArch()
		arch.TieWordEmbeddings = &tc.declaredTied
		if _, err := Assemble(tensors, arch, names); err == nil {
			t.Fatalf("declared tied %v accepted head presence %v", tc.declaredTied, tc.includeHead)
		}
	}
}

func boolPointer(value bool) *bool { return &value }
