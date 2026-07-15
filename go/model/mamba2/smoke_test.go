// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import (
	"os"
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

// TestMamba2RealCheckpointSmoke is the cross-engine correctness gate: load a REAL HF Mamba2ForCausalLM
// checkpoint through the native loader and greedy-decode, then compare token-for-token to the HF
// transformers reference (run separately). This is what the synthetic carry-invariant tests cannot prove —
// that the loader's weight-name resolution, the geometry-from-shapes, and the recurrence numerics produce
// CORRECT tokens on real weights. Env-guarded (MAMBA2_SMOKE_DIR) so it is not part of the normal suite; a
// functional smoke on a real model, not a unit test (run on request).
//
// Reference (AntonV/mamba2-130m-hf, prompt "The capital of France is", greedy 12):
//
//	HF GEN_IDS = [247 2846 273 253 5112 952 13 285 253 5347 273 253]  ("a city of the French people, and the capital of the")
func TestMamba2RealCheckpointSmoke(t *testing.T) {
	dir := os.Getenv("MAMBA2_SMOKE_DIR")
	if dir == "" {
		t.Skip("set MAMBA2_SMOKE_DIR to a real HF mamba2 checkpoint dir")
	}
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		t.Fatalf("load safetensors: %v", err)
	}
	defer func() { _ = dm.Close() }()

	m, err := LoadMambaModel(dm.Tensors, 1e-5) // mamba2 default layer_norm_epsilon
	if err != nil {
		t.Fatalf("LoadMambaModel: %v", err)
	}
	t.Logf("loaded: %d layers, D=%d, vocab=%d, cfg=%+v", len(m.Layers), m.D, m.Vocab, m.Cfg)

	prompt := []int32{510, 5347, 273, 6181, 310} // "The capital of France is"

	// Diagnostic: native first-token top-5 vs the HF reference, to localise prefill-bug vs decode-bug.
	dsess := NewSession(m)
	hid, derr := dsess.Forward(prompt)
	if derr != nil {
		t.Fatalf("prefill: %v", derr)
	}
	lg := dsess.headLogits(hid[(len(prompt)-1)*m.D:])
	ids, vals := top5(lg)
	t.Logf("native first-token top5 IDs=%v vals=%v", ids, vals)
	t.Logf("HF     first-token top5 IDs=[247 275 253 327 417] vals=[-9.97 -10.101 -10.453 -10.732 -11.012]")

	want := []int32{247, 2846, 273, 253, 5112, 952, 13, 285, 253, 5347, 273, 253}
	gen, err := NewSession(m).Generate(prompt, len(want), -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	t.Logf("native GEN_IDS = %v", gen)
	mism := 0
	for i := range want {
		if i >= len(gen) || gen[i] != want[i] {
			mism++
		}
	}
	if mism > 0 {
		t.Fatalf("native diverged from HF reference in %d/%d tokens\n native %v\n HF     %v", mism, len(want), gen, want)
	}
	t.Logf("✓ native mamba2 == HF transformers token-for-token (%d tokens) — real-checkpoint correctness confirmed", len(want))
}

func top5(v []float32) ([]int, []float32) {
	idx := make([]int, len(v))
	for i := range idx {
		idx[i] = i
	}
	for i := 0; i < 5 && i < len(idx); i++ {
		best := i
		for j := i + 1; j < len(idx); j++ {
			if v[idx[j]] > v[idx[best]] {
				best = j
			}
		}
		idx[i], idx[best] = idx[best], idx[i]
	}
	ids := make([]int, 5)
	vals := make([]float32, 5)
	for i := range 5 {
		ids[i] = idx[i]
		vals[i] = v[idx[i]]
	}
	return ids, vals
}
