// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// TestDumpArchFromSnapshot is an env-gated real-config diagnostic: GEMMA4_SNAP=<snapshot dir>
// parses config.json through the SAME lane the engine loader uses (parseGemma4Config → Arch)
// and dumps every attention-geometry field plus the first global/sliding layer specs — the
// one-print check that a derived arch matches the config's declared truth.
func TestDumpArchFromSnapshot(t *testing.T) {
	snap := os.Getenv("GEMMA4_SNAP")
	if snap == "" {
		t.Skip("GEMMA4_SNAP not set")
	}
	raw := core.ReadFile(snap + "/config.json")
	if !raw.OK {
		t.Fatalf("read config: %v", raw.Value)
	}
	cfg, err := parseGemma4Config(raw.Value.([]byte))
	if err != nil {
		t.Fatalf("parseGemma4Config: %v", err)
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	t.Logf("Hidden=%d Layer=%d Heads=%d KVHeads=%d GlobalKVHeads=%d HeadDim=%d GlobalHeadDim=%d FF=%d",
		arch.Hidden, len(arch.Layer), arch.Heads, arch.KVHeads, arch.GlobalKVHeads, arch.HeadDim, arch.GlobalHeadDim, arch.FF)
	t.Logf("SlidingWindow=%d RotaryDim=%d RotaryDimLocal=%d RopeBase=%.0f RopeLocalBase=%.0f ValueNorm=%v Eps=%g",
		arch.SlidingWindow, arch.RotaryDim, arch.RotaryDimLocal, arch.RopeBase, arch.RopeLocalBase, arch.ValueNorm, arch.Eps)
	var firstG, firstS = -1, -1
	for li, sp := range arch.Layer {
		if sp.Attention == model.GlobalAttention && firstG < 0 {
			firstG = li
		}
		if sp.Attention != model.GlobalAttention && firstS < 0 {
			firstS = li
		}
		if firstG >= 0 && firstS >= 0 {
			break
		}
	}
	for _, li := range []int{firstS, firstG} {
		if li < 0 {
			continue
		}
		sp := arch.Layer[li]
		t.Logf("L%02d attn=%v KVHeads=%d HeadDim=%d KVShareFrom=%d OwnsCache=%v MoE=%v",
			li, sp.Attention, sp.KVHeads, sp.HeadDim, sp.KVShareFrom, sp.OwnsCache(), sp.MoE)
	}
}
