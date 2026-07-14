// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"strings"
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

func TestAssemble_EmbeddingProjection_Good(t *testing.T) {
	tensors := minimalDenseTensors("BF16")
	tensors["embed.weight"] = safetensors.Tensor{Shape: []int{8, 2}, Data: make([]byte, 8*2*2)}
	tensors["project_in.weight"] = safetensors.Tensor{Shape: []int{4, 2}, Data: make([]byte, 4*2*2)}
	tensors["project_out.weight"] = safetensors.Tensor{Shape: []int{2, 4}, Data: make([]byte, 2*4*2)}
	names := minimalDenseNames()
	names.EmbedProjectionIn, names.EmbedProjectionOut = "project_in", "project_out"
	arch := minimalDenseArch()
	arch.EmbeddingDim = 2
	m, err := Assemble(tensors, arch, names)
	if err != nil {
		t.Fatal(err)
	}
	if m.Embed.InDim != 2 || m.EmbedProjectionIn.InDim != 2 || m.EmbedProjectionOut.InDim != 4 {
		t.Fatalf("embedding widths: embed=%d project_in=%d project_out=%d", m.Embed.InDim, m.EmbedProjectionIn.InDim, m.EmbedProjectionOut.InDim)
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

// kEqVNames extends the minimal dense names with a value-projection suffix, so a fixture can choose
// to carry or omit the v_proj — the input to Assemble's per-layer K==V op-selection resolution.
func kEqVNames() WeightNames {
	n := minimalDenseNames()
	n.V = ".v"
	return n
}

// TestAssemble_AttentionKEqV_Good: a layer that CARRIES a v_proj weight is not K==V — Assemble
// declares LayerSpec.AttentionKEqV false, so a backend records a separate value projection.
func TestAssemble_AttentionKEqV_Good(t *testing.T) {
	tensors := minimalDenseTensors("BF16")
	tensors["layer.0.v.weight"] = safetensors.Tensor{Shape: []int{4, 4}, Data: make([]byte, 4*4*2)}
	m, err := Assemble(tensors, minimalDenseArch(), kEqVNames())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if m.Layers[0].V == nil {
		t.Fatal("the v_proj weight should have loaded")
	}
	if m.Arch.Layer[0].AttentionKEqV {
		t.Fatal("a layer carrying a v_proj must not be declared K==V")
	}
}

// TestAssemble_AttentionKEqV_Bad: a cache-owning layer with NO v_proj weight IS K==V — Assemble
// declares LayerSpec.AttentionKEqV true, so a backend routes the value through the key projection
// (gemma4's global_attention layers). minimalDenseNames carries no V suffix, so no v_proj loads.
func TestAssemble_AttentionKEqV_Bad(t *testing.T) {
	m, err := Assemble(minimalDenseTensors("BF16"), minimalDenseArch(), minimalDenseNames())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if m.Layers[0].V != nil {
		t.Fatal("no v_proj weight should have loaded")
	}
	if !m.Arch.Layer[0].AttentionKEqV {
		t.Fatal("a cache-owning layer with no v_proj must be declared K==V")
	}
}

// TestAssemble_AttentionKEqV_Ugly: a KV-SHARED layer carries no v_proj of its own (it attends the
// owner's cache), so it too resolves to AttentionKEqV true — even alongside an owner that DOES carry
// a v_proj (false). Proves the resolution is per-layer and driven by weight presence, not a single
// whole-model flag.
func TestAssemble_AttentionKEqV_Ugly(t *testing.T) {
	vec := func(n int) safetensors.Tensor {
		return safetensors.Tensor{Shape: []int{n}, Data: make([]byte, n*2), Dtype: "BF16"}
	}
	mat := func(rows, cols int) safetensors.Tensor {
		return safetensors.Tensor{Shape: []int{rows, cols}, Data: make([]byte, rows*cols*2)}
	}
	tensors := map[string]safetensors.Tensor{
		"embed.weight": mat(8, 4), "norm.weight": vec(4),
		// layer 0 OWNS its cache and carries q/k/v/o.
		"layer.0.attn_norm.weight": vec(4),
		"layer.0.q.weight":         mat(4, 4), "layer.0.k.weight": mat(4, 4),
		"layer.0.v.weight": mat(4, 4), "layer.0.o.weight": mat(4, 4),
		"layer.0.mlp_norm.weight": vec(4),
		"layer.0.gate.weight":     mat(8, 4), "layer.0.up.weight": mat(8, 4), "layer.0.down.weight": mat(4, 8),
		// layer 1 SHARES layer 0's cache — no k/v of its own.
		"layer.1.attn_norm.weight": vec(4),
		"layer.1.q.weight":         mat(4, 4), "layer.1.o.weight": mat(4, 4),
		"layer.1.mlp_norm.weight":  vec(4),
		"layer.1.gate.weight":      mat(8, 4), "layer.1.up.weight": mat(8, 4), "layer.1.down.weight": mat(4, 8),
	}
	arch := minimalDenseArch()
	arch.Layer = []LayerSpec{
		{Attention: GlobalAttention, CacheIndex: 0, KVShareFrom: 0, HeadDim: 2, KVHeads: 2},
		{Attention: GlobalAttention, CacheIndex: -1, KVShareFrom: 0, HeadDim: 2, KVHeads: 2},
	}
	m, err := Assemble(tensors, arch, kEqVNames())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if m.Arch.Layer[0].AttentionKEqV {
		t.Fatal("owner layer 0 carries a v_proj — must not be declared K==V")
	}
	if !m.Arch.Layer[1].AttentionKEqV {
		t.Fatal("KV-shared layer 1 carries no v_proj — must be declared K==V")
	}
}

// normOpNames extends the minimal dense names with the four norm-op suffixes so
// their weights load when the tensors carry them.
func normOpNames() WeightNames {
	n := minimalDenseNames()
	n.QNorm = ".q_norm.weight"
	n.KNorm = ".k_norm.weight"
	n.PostAttnNorm = ".post_attn_norm.weight"
	n.PostFFNorm = ".post_ff_norm.weight"
	return n
}

// TestAssemble_NormOpSelections_Good: a checkpoint carrying all four norm weights has every
// norm-op selection declared true — backends bind the declared selections (#57 slice 3)
// instead of re-probing weight buffers.
func TestAssemble_NormOpSelections_Good(t *testing.T) {
	vec := func(n int) safetensors.Tensor {
		return safetensors.Tensor{Shape: []int{n}, Data: make([]byte, n*2), Dtype: "BF16"}
	}
	tensors := minimalDenseTensors("BF16")
	tensors["layer.0.q_norm.weight"] = vec(4)
	tensors["layer.0.k_norm.weight"] = vec(4)
	tensors["layer.0.post_attn_norm.weight"] = vec(4)
	tensors["layer.0.post_ff_norm.weight"] = vec(4)
	m, err := Assemble(tensors, minimalDenseArch(), normOpNames())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	spec := m.Arch.Layer[0]
	if !spec.AttentionQNorm || !spec.AttentionKNorm || !spec.PostAttnNorm || !spec.PostFFNorm {
		t.Fatalf("norm-op selections = %+v, want all true", spec)
	}
}

// TestAssemble_NormOpSelections_Bad: a checkpoint with NO norm weights declares every norm-op
// selection false — an absent weight can never select an op.
func TestAssemble_NormOpSelections_Bad(t *testing.T) {
	m, err := Assemble(minimalDenseTensors("BF16"), minimalDenseArch(), minimalDenseNames())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	spec := m.Arch.Layer[0]
	if spec.AttentionQNorm || spec.AttentionKNorm || spec.PostAttnNorm || spec.PostFFNorm {
		t.Fatalf("norm-op selections = %+v, want all false", spec)
	}
}

// TestAssemble_LayerScalar_Good: a checkpoint carrying a per-layer output scalar weight
// (gemma4 diffusion's .layer_scalar) has the op selection declared true — backends bind
// the declared selection (#57) instead of re-probing weight buffers.
func TestAssemble_LayerScalar_Good(t *testing.T) {
	tensors := minimalDenseTensors("BF16")
	tensors["layer.0.layer_scalar"] = safetensors.Tensor{Shape: []int{1}, Data: make([]byte, 2), Dtype: "BF16"}
	names := minimalDenseNames()
	names.LayerScalar = ".layer_scalar"
	m, err := Assemble(tensors, minimalDenseArch(), names)
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if !m.Arch.Layer[0].LayerScalar {
		t.Fatal("layer_scalar weight present, selection must be declared")
	}
}

// TestAssemble_LayerScalar_Bad: a checkpoint with NO layer-scalar weight declares the
// selection false — an absent weight can never select an op.
func TestAssemble_LayerScalar_Bad(t *testing.T) {
	names := minimalDenseNames()
	names.LayerScalar = ".layer_scalar"
	m, err := Assemble(minimalDenseTensors("BF16"), minimalDenseArch(), names)
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if m.Arch.Layer[0].LayerScalar {
		t.Fatal("no layer_scalar weight, selection must not be declared")
	}
}

// TestAssemble_LayerScalar_Ugly: unlike the stack-uniform norm selections, the scalar may
// sit on a SUBSET of layers — declaration is per layer, exactly following weight presence
// (backends bind the ×1.0 identity for undeclared layers to keep the op layout uniform).
func TestAssemble_LayerScalar_Ugly(t *testing.T) {
	tensors := minimalDenseTensors("BF16")
	for name, tt := range tensors { // clone layer.0's weights as layer.1
		if after, ok := strings.CutPrefix(name, "layer.0."); ok {
			tensors["layer.1."+after] = tt
		}
	}
	tensors["layer.1.layer_scalar"] = safetensors.Tensor{Shape: []int{1}, Data: make([]byte, 2), Dtype: "BF16"}
	arch := minimalDenseArch()
	arch.Layer = append(arch.Layer, LayerSpec{Attention: GlobalAttention, CacheIndex: 1, KVShareFrom: 1, HeadDim: 2, KVHeads: 2})
	names := minimalDenseNames()
	names.LayerScalar = ".layer_scalar"
	m, err := Assemble(tensors, arch, names)
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if m.Arch.Layer[0].LayerScalar {
		t.Fatal("layer 0 carries no scalar weight, must not declare")
	}
	if !m.Arch.Layer[1].LayerScalar {
		t.Fatal("layer 1 carries the scalar weight, must declare")
	}
}

// TestAssemble_NormOpSelections_Ugly: the selections are independent per norm — a checkpoint
// carrying q_norm but no k_norm (the name is mapped, the tensor is absent) declares exactly
// the present one. Proves declaration follows weight presence, not the name mapping.
func TestAssemble_NormOpSelections_Ugly(t *testing.T) {
	vec := func(n int) safetensors.Tensor {
		return safetensors.Tensor{Shape: []int{n}, Data: make([]byte, n*2), Dtype: "BF16"}
	}
	tensors := minimalDenseTensors("BF16")
	tensors["layer.0.q_norm.weight"] = vec(4)
	m, err := Assemble(tensors, minimalDenseArch(), normOpNames())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	spec := m.Arch.Layer[0]
	if !spec.AttentionQNorm {
		t.Fatal("q_norm weight present, selection must be declared")
	}
	if spec.AttentionKNorm || spec.PostAttnNorm || spec.PostFFNorm {
		t.Fatalf("absent norms declared: %+v", spec)
	}
}
