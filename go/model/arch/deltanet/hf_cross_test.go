// SPDX-Licence-Identifier: EUPL-1.2

package deltanet

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// TestGatedDeltaVsHFReference cross-checks the native gated delta-rule recurrence against HF transformers'
// torch_recurrent_gated_delta_rule (Qwen3-Next) on identical random inputs — the layer-level correctness
// gate the mamba2 smoke showed the synthetic carry tests cannot give. The HF reference is dumped by a
// Python script to /tmp/gd_ref.json (q/k/v/β/g + output); this loads it, applies the same l2norm(q) the
// block does, runs GatedDeltaRuleF32 (which l2-norms k + scales q by 1/√D internally, matching HF's
// use_qk_l2norm_in_kernel), and compares. Env/file-guarded; not part of the normal suite.
func TestGatedDeltaVsHFReference(t *testing.T) {
	data, err := os.ReadFile("/tmp/gd_ref.json")
	if err != nil {
		t.Skip("no /tmp/gd_ref.json — run the HF dump first")
	}
	var ref struct {
		H, D, L               int
		Q, K, V, Beta, G, Out []float32
	}
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatalf("parse ref: %v", err)
	}
	H, D, L := ref.H, ref.D, ref.L

	// l2norm(q) per (token,head) over D — HF l2-norms q inside the kernel; the native block does it before
	// the recurrence, so we replicate that here.
	qn := make([]float32, len(ref.Q))
	for row := 0; row < L*H; row++ {
		var ss float64
		for i := range D {
			qv := float64(ref.Q[row*D+i])
			ss += qv * qv
		}
		inv := 1.0 / math.Sqrt(ss+1e-6)
		for i := range D {
			qn[row*D+i] = float32(float64(ref.Q[row*D+i]) * inv)
		}
	}
	alpha := make([]float32, len(ref.G)) // α = exp(g) (g is the per-token log-decay)
	for i, gv := range ref.G {
		alpha[i] = float32(math.Exp(float64(gv)))
	}

	o, _, err := GatedDeltaRuleF32(qn, ref.K, ref.V, ref.Beta, alpha, nil, L, H, D, float32(1.0/math.Sqrt(float64(D))), 1e-6)
	if err != nil {
		t.Fatalf("GatedDeltaRuleF32: %v", err)
	}
	var maxRel float64
	for i := range ref.Out {
		d := math.Abs(float64(o[i] - ref.Out[i]))
		rel := d / (1 + math.Abs(float64(ref.Out[i])))
		if rel > maxRel {
			maxRel = rel
		}
	}
	if maxRel > 1e-3 {
		t.Fatalf("native recurrence diverged from HF Qwen3-Next: maxRel=%.3e\n native[:3]=%v\n HF[:3]    =%v", maxRel, o[:3], ref.Out[:3])
	}
	t.Logf("✓ native GatedDeltaRuleF32 == HF torch_recurrent_gated_delta_rule (maxRel %.2e over %d elems) — gated-delta recurrence correct vs Qwen3-Next", maxRel, len(ref.Out))
}
